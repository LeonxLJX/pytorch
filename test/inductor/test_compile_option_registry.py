# Owner(s): ["module: inductor"]
import contextlib
import unittest
from unittest import mock

import torch
import torch.fx as fx
from torch._inductor import compile_fx_ext, compile_option_registry, config
from torch._inductor.compile_fx import (
    compile_fx,
    FxCompileMode,
    get_patched_config_dict,
)
from torch._inductor.compile_option_registry import (
    CompileOptionRoute,
    get_compile_option_route,
    patch_compile_options,
    patch_routed_configs,
    register_compile_option,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal import fake_config_module
from torch.utils._import_utils import import_dill


dill = import_dill()
HAS_DILL = dill is not None


FAKE_MODULE = "torch.testing._internal.fake_config_module"
FAKE_MODULE2 = "torch.testing._internal.fake_config_module2"


def dummy_fn(x):
    return torch.sigmoid(x + 1.0) / 10.0


@contextlib.contextmanager
def fake_backend_routes():
    with mock.patch.dict(compile_option_registry._routes):
        register_compile_option("fake_backend_bool", module=FAKE_MODULE, key="e_bool")
        register_compile_option(
            "fake_backend2_bool", module=FAKE_MODULE2, key="e_aliasing_bool"
        )
        yield


class TestCompileOptionRegistry(TestCase):
    def test_register_and_lookup(self):
        with fake_backend_routes():
            route = CompileOptionRoute(module=FAKE_MODULE, key="e_bool")
            self.assertEqual(get_compile_option_route("fake_backend_bool"), route)
            # dashed spelling resolves to the same route
            self.assertEqual(get_compile_option_route("fake-backend-bool"), route)
        self.assertIsNone(get_compile_option_route("fake_backend_bool"))

    def test_register_with_module_object(self):
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=fake_config_module, key="e_bool"
            )
            self.assertEqual(
                get_compile_option_route("fake_backend_bool"),
                CompileOptionRoute(module=FAKE_MODULE, key="e_bool"),
            )
            # targets are validated eagerly when the module object is passed
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                register_compile_option(
                    "bad_key", module=fake_config_module, key="missing_key"
                )

    def test_register_duplicate(self):
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=FAKE_MODULE, key="e_bool"
            )
            # re-registering the same target is a no-op (backends may reload)
            register_compile_option(
                "fake-backend-bool", module=FAKE_MODULE, key="e_bool"
            )
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                register_compile_option(
                    "fake_backend_bool", module=FAKE_MODULE, key="e_string"
                )

    def test_register_invalid_name(self):
        with mock.patch.dict(compile_option_registry._routes):
            for name in ("not an identifier", "1abc", "class", "a.b", "\u212a"):
                with self.assertRaises(AssertionError):
                    register_compile_option(name, module=FAKE_MODULE, key="e_bool")

    def test_register_rejects_shadowing_inductor_config(self):
        with mock.patch.dict(compile_option_registry._routes):
            with self.assertRaisesRegex(AssertionError, "shadows"):
                register_compile_option(
                    "max_fusion_size", module=FAKE_MODULE, key="e_bool"
                )

    def test_apply_options_routed(self):
        with fake_backend_routes():
            wrapper = torch._TorchCompileInductorWrapper(
                None, {"fake_backend_bool": False, "max_fusion_size": 33}, None
            )
            self.assertEqual(wrapper.config["fake_backend_bool"], False)
            self.assertEqual(wrapper.config["max_fusion_size"], 33)
            self.assertEqual(
                wrapper._config_routes,
                {
                    "fake_backend_bool": CompileOptionRoute(
                        module=FAKE_MODULE, key="e_bool"
                    )
                },
            )
            # dashed spelling is normalized like inductor's own options
            wrapper = torch._TorchCompileInductorWrapper(
                None, {"fake-backend-bool": False}, None
            )
            self.assertEqual(wrapper.config, {"fake_backend_bool": False})
            # type is checked against the owning module
            with self.assertRaisesRegex(RuntimeError, "Unexpected type of attr"):
                torch._TorchCompileInductorWrapper(
                    None, {"fake_backend_bool": "not a bool"}, None
                )

    def test_apply_options_unknown_name_rejected(self):
        with fake_backend_routes():
            with self.assertRaisesRegex(RuntimeError, "Unexpected optimization option"):
                torch._TorchCompileInductorWrapper(
                    None, {"unregistered_option": True}, None
                )

    def test_apply_options_rejects_bad_route(self):
        with mock.patch.dict(compile_option_registry._routes):
            register_compile_option(
                "fake_backend_bool", module=FAKE_MODULE, key="missing_key"
            )
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                torch._TorchCompileInductorWrapper(
                    None, {"fake_backend_bool": True}, None
                )

            register_compile_option(
                "fake_backend_bool2", module="torch._inductor.compile_fx", key="e_bool"
            )
            with self.assertRaisesRegex(RuntimeError, "not a ConfigModule"):
                torch._TorchCompileInductorWrapper(
                    None, {"fake_backend_bool2": True}, None
                )

    def test_patch_compile_options(self):
        default_fusion_size = config.max_fusion_size
        with fake_backend_routes():
            route = get_compile_option_route("fake_backend_bool")
            with patch_compile_options(
                {"fake_backend_bool": False, "max_fusion_size": 64},
                {"fake_backend_bool": route},
            ):
                self.assertFalse(fake_config_module.e_bool)
                self.assertEqual(config.max_fusion_size, 64)
            self.assertTrue(fake_config_module.e_bool)
            self.assertEqual(config.max_fusion_size, default_fusion_size)

    def test_patch_compile_options_decorator_reentry(self):
        # backwards is compiled out of scope of the forward patch context; the
        # decorator form must re-enter every owner patch on each call
        with fake_backend_routes():
            route = get_compile_option_route("fake_backend_bool")
            observed = []

            def inner(x):
                observed.append((fake_config_module.e_bool, config.max_fusion_size))
                return x

            decorated = patch_compile_options(
                {"fake_backend_bool": False, "max_fusion_size": 64},
                {"fake_backend_bool": route},
            )(inner)
        decorated(torch.zeros(1))
        self.assertEqual(observed, [(False, 64)])

    def test_patch_compile_options_reentrant(self):
        # the same instance must be safe to enter while already active, like
        # config.patch (nested graph compilation, or backward compiling on
        # another thread)
        with fake_backend_routes():
            route = get_compile_option_route("fake_backend_bool")
            patcher = patch_compile_options(
                {"fake_backend_bool": False}, {"fake_backend_bool": route}
            )
            with patcher:
                self.assertFalse(fake_config_module.e_bool)
                with patcher:
                    self.assertFalse(fake_config_module.e_bool)
                self.assertFalse(fake_config_module.e_bool)
            self.assertTrue(fake_config_module.e_bool)

    def test_compile_fx_patches_owner_module(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2)

        gm = fx.symbolic_trace(M())
        x = torch.randn(4)
        observed = []

        def inner_compile(gm_, *args, **kwargs):
            observed.append((fake_config_module.e_bool, config.max_fusion_size))
            return gm_

        with fake_backend_routes():
            compiled = compile_fx(
                gm,
                [x],
                inner_compile=inner_compile,
                config_patches={"fake_backend_bool": False, "max_fusion_size": 64},
                config_patch_routes={
                    "fake_backend_bool": get_compile_option_route("fake_backend_bool")
                },
            )
        compiled(x)
        self.assertEqual(observed, [(False, 64)])
        # patches are undone once compilation is over
        self.assertTrue(fake_config_module.e_bool)

    def test_compile_fx_backward_sees_routed_patches(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2) * x.cos()

        gm = fx.symbolic_trace(M())
        x = torch.randn(4, requires_grad=True)
        observed = []

        def inner_compile(gm_, *args, **kwargs):
            observed.append(fake_config_module.e_bool)
            return gm_

        with fake_backend_routes():
            compiled = compile_fx(
                gm,
                [x],
                inner_compile=inner_compile,
                config_patches={"fake_backend_bool": False},
                config_patch_routes={
                    "fake_backend_bool": get_compile_option_route("fake_backend_bool")
                },
            )
            out = compiled(x)
        out.sum().backward()
        # backward compiles out of scope of the forward patch context (and in
        # some configurations only when backward runs), but the re-entered
        # decorator keeps the owner module patched for every compilation
        self.assertGreaterEqual(len(observed), 2)
        self.assertTrue(all(value is False for value in observed))
        self.assertTrue(fake_config_module.e_bool)

    def test_patch_routed_configs(self):
        with patch_routed_configs({FAKE_MODULE: {"e_bool": False}}):
            self.assertFalse(fake_config_module.e_bool)
        self.assertTrue(fake_config_module.e_bool)

    @unittest.skipUnless(HAS_DILL, "dill not available")
    def test_compile_fx_ext_replays_routed_configs(self):
        # subprocess compile workers must see the routed values, not the
        # backend's defaults; SERIALIZE mode runs the same serialize ->
        # _run_in_child -> deserialize path in-process
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x * 2)

        gm = fx.symbolic_trace(M())
        x = torch.randn(4)
        snapshots_seen = []
        child_values = []
        real_patch_routed = compile_option_registry.patch_routed_configs

        def spy(snapshots):
            snapshots_seen.append(snapshots)
            patcher = real_patch_routed(snapshots)

            @contextlib.contextmanager
            def recording():
                with patcher:
                    child_values.append(fake_config_module.e_bool)
                    yield

            return recording()

        with fake_backend_routes():
            with (
                mock.patch(
                    "torch._inductor.compile_fx.fx_compile_mode",
                    FxCompileMode.SERIALIZE,
                ),
                mock.patch.object(compile_fx_ext, "patch_routed_configs", spy),
            ):
                compiled = compile_fx(
                    gm,
                    [x],
                    config_patches={"fake_backend_bool": False},
                    config_patch_routes={
                        "fake_backend_bool": get_compile_option_route(
                            "fake_backend_bool"
                        )
                    },
                )
                compiled(x)

        # only modules imported at serialize time are snapshotted; the second
        # registered owner was never used
        self.assertEqual(set(snapshots_seen[0]), {FAKE_MODULE})
        self.assertEqual(snapshots_seen[0][FAKE_MODULE]["e_bool"], False)
        self.assertEqual(child_values, [False])
        self.assertTrue(fake_config_module.e_bool)

    def test_get_patched_config_dict_routed(self):
        with fake_backend_routes():
            result = get_patched_config_dict(
                {
                    "fake_backend_bool": False,
                    "fake_backend2_bool": True,
                    "max_fusion_size": 64,
                },
                {
                    "fake_backend_bool": CompileOptionRoute(
                        module=FAKE_MODULE, key="e_bool"
                    ),
                    "fake_backend2_bool": CompileOptionRoute(
                        module=FAKE_MODULE2, key="e_aliasing_bool"
                    ),
                },
            )
        self.assertEqual(result["fake_backend_bool"], False)
        self.assertEqual(result["fake_backend2_bool"], True)
        self.assertEqual(result["max_fusion_size"], 64)

    def test_torch_compile_routed_option(self):
        with fake_backend_routes():
            optimized = torch.compile(dummy_fn, options={"fake_backend_bool": False})
            # get_compiler_config runs at torch.compile() time, before any
            # device is known; routed keys must not crash it and must surface
            # their effective value
            compiler_config = optimized.get_compiler_config()
        self.assertEqual(compiler_config["fake_backend_bool"], False)
        # compilation must not depend on the registry still holding the route:
        # the wrapper resolved and retained it at apply_options time
        x = torch.randn(10)
        torch.testing.assert_close(optimized(x), dummy_fn(x))
        # the owner module is restored once compilation is over
        self.assertTrue(fake_config_module.e_bool)


if __name__ == "__main__":
    run_tests()
