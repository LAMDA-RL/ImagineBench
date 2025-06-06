import re
from collections import OrderedDict

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerBasketballEnvV2,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnvV2,
    SawyerButtonPressEnvV2,
    SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2,
    SawyerCoffeePullEnvV2,
    SawyerCoffeePushEnvV2,
    SawyerDialTurnEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
    SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerFaucetCloseEnvV2,
    SawyerFaucetOpenEnvV2,
    SawyerHammerEnvV2,
    SawyerHandInsertEnvV2,
    SawyerHandlePressEnvV2,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePullEnvV2,
    SawyerHandlePullSideEnvV2,
    SawyerLeverPullEnvV2,
    SawyerNutAssemblyEnvV2,
    SawyerNutDisassembleEnvV2,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnvV2,
    SawyerPickOutOfHoleEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideEnvV2,
    SawyerPlateSlideSideEnvV2,
    SawyerPushBackEnvV2,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnvV2,
    SawyerSoccerEnvV2,
    SawyerStickPullEnvV2,
    SawyerStickPushEnvV2,
    SawyerSweepEnvV2,
    SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnvV2,
    # Newly added
    SawyerMakeCoffeeEnvV2,
    SawyerLockedDoorOpenEnvV2,
)

ALL_V2_ENVIRONMENTS = OrderedDict(
    (
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("basketball-v2", SawyerBasketballEnvV2),
        ("bin-picking-v2", SawyerBinPickingEnvV2),
        ("box-close-v2", SawyerBoxCloseEnvV2),
        ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("button-press-v2", SawyerButtonPressEnvV2),
        ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
        ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
        ("coffee-pull-v2", SawyerCoffeePullEnvV2),
        ("coffee-push-v2", SawyerCoffeePushEnvV2),
        ("dial-turn-v2", SawyerDialTurnEnvV2),
        ("disassemble-v2", SawyerNutDisassembleEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
        ("door-lock-v2", SawyerDoorLockEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("door-unlock-v2", SawyerDoorUnlockEnvV2),
        ("hand-insert-v2", SawyerHandInsertEnvV2),
        ("drawer-close-v2", SawyerDrawerCloseEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("faucet-open-v2", SawyerFaucetOpenEnvV2),
        ("faucet-close-v2", SawyerFaucetCloseEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
        ("handle-press-v2", SawyerHandlePressEnvV2),
        ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
        ("handle-pull-v2", SawyerHandlePullEnvV2),
        ("lever-pull-v2", SawyerLeverPullEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("push-back-v2", SawyerPushBackEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("plate-slide-v2", SawyerPlateSlideEnvV2),
        ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
        ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
        ("soccer-v2", SawyerSoccerEnvV2),
        ("stick-push-v2", SawyerStickPushEnvV2),
        ("stick-pull-v2", SawyerStickPullEnvV2),
        ("push-wall-v2", SawyerPushWallEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("reach-wall-v2", SawyerReachWallEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("shelf-place-v2", SawyerShelfPlaceEnvV2),
        ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
        ("sweep-v2", SawyerSweepEnvV2),
        ("window-open-v2", SawyerWindowOpenEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
        # Hard-Level
        ("make-coffee-v2", SawyerMakeCoffeeEnvV2),
        ("locked-door-open-v2", SawyerLockedDoorOpenEnvV2),
        # Rephrase-Level
        ('rep-reach-v2', SawyerReachEnvV2),
        ('rep-push-v2', SawyerPushEnvV2),
        ('rep-pick-place-v2', SawyerPickPlaceEnvV2),
        ('rep-button-press-v2', SawyerButtonPressEnvV2),
        ('rep-door-unlock-v2', SawyerDoorUnlockEnvV2),
        ('rep-door-open-v2', SawyerDoorEnvV2),
        ('rep-window-open-v2', SawyerWindowOpenEnvV2),
        ('rep-faucet-open-v2', SawyerFaucetOpenEnvV2),
        ('rep-coffee-push-v2', SawyerCoffeePushEnvV2),
        ('rep-coffee-button-v2', SawyerCoffeeButtonEnvV2),
    )
)


_NUM_METAWORLD_ENVS = len(ALL_V2_ENVIRONMENTS)
# V2 DICTS

MT10_V2 = OrderedDict(
    (
        ("reach-v2", SawyerReachEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("drawer-close-v2", SawyerDrawerCloseEnvV2),
        ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("window-open-v2", SawyerWindowOpenEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
    ),
)


MT10_V2_ARGS_KWARGS = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT10_V2.items()
}

ML10_V2 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("reach-v2", SawyerReachEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    ("door-open-v2", SawyerDoorEnvV2),
                    ("drawer-close-v2", SawyerDrawerCloseEnvV2),
                    ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("window-open-v2", SawyerWindowOpenEnvV2),
                    ("sweep-v2", SawyerSweepEnvV2),
                    ("basketball-v2", SawyerBasketballEnvV2),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("drawer-open-v2", SawyerDrawerOpenEnvV2),
                    ("door-close-v2", SawyerDoorCloseEnvV2),
                    ("shelf-place-v2", SawyerShelfPlaceEnvV2),
                    ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
                    (
                        "lever-pull-v2",
                        SawyerLeverPullEnvV2,
                    ),
                )
            ),
        ),
    )
)


ml10_train_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML10_V2["train"].items()
}

ml10_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML10_V2["test"].items()
}

ML10_ARGS_KWARGS = dict(
    train=ml10_train_args_kwargs,
    test=ml10_test_args_kwargs,
)

ML1_V2 = OrderedDict((("train", ALL_V2_ENVIRONMENTS), ("test", ALL_V2_ENVIRONMENTS)))

ML1_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML1_V2["train"].items()
}
MT50_V2 = OrderedDict(
    (
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("basketball-v2", SawyerBasketballEnvV2),
        ("bin-picking-v2", SawyerBinPickingEnvV2),
        ("box-close-v2", SawyerBoxCloseEnvV2),
        ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("button-press-v2", SawyerButtonPressEnvV2),
        ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
        ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
        ("coffee-pull-v2", SawyerCoffeePullEnvV2),
        ("coffee-push-v2", SawyerCoffeePushEnvV2),
        ("dial-turn-v2", SawyerDialTurnEnvV2),
        ("disassemble-v2", SawyerNutDisassembleEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
        ("door-lock-v2", SawyerDoorLockEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("door-unlock-v2", SawyerDoorUnlockEnvV2),
        ("hand-insert-v2", SawyerHandInsertEnvV2),
        ("drawer-close-v2", SawyerDrawerCloseEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("faucet-open-v2", SawyerFaucetOpenEnvV2),
        ("faucet-close-v2", SawyerFaucetCloseEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
        ("handle-press-v2", SawyerHandlePressEnvV2),
        ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
        ("handle-pull-v2", SawyerHandlePullEnvV2),
        ("lever-pull-v2", SawyerLeverPullEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("push-back-v2", SawyerPushBackEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("plate-slide-v2", SawyerPlateSlideEnvV2),
        ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
        ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
        ("soccer-v2", SawyerSoccerEnvV2),
        ("stick-push-v2", SawyerStickPushEnvV2),
        ("stick-pull-v2", SawyerStickPullEnvV2),
        ("push-wall-v2", SawyerPushWallEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("reach-wall-v2", SawyerReachWallEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("shelf-place-v2", SawyerShelfPlaceEnvV2),
        ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
        ("sweep-v2", SawyerSweepEnvV2),
        ("window-open-v2", SawyerWindowOpenEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
    )
)

MT50_V2_ARGS_KWARGS = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT50_V2.items()
}

ML45_V2 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("assembly-v2", SawyerNutAssemblyEnvV2),
                    ("basketball-v2", SawyerBasketballEnvV2),
                    ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
                    ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
                    ("button-press-v2", SawyerButtonPressEnvV2),
                    ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
                    ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
                    ("coffee-pull-v2", SawyerCoffeePullEnvV2),
                    ("coffee-push-v2", SawyerCoffeePushEnvV2),
                    ("dial-turn-v2", SawyerDialTurnEnvV2),
                    ("disassemble-v2", SawyerNutDisassembleEnvV2),
                    ("door-close-v2", SawyerDoorCloseEnvV2),
                    ("door-open-v2", SawyerDoorEnvV2),
                    ("drawer-close-v2", SawyerDrawerCloseEnvV2),
                    ("drawer-open-v2", SawyerDrawerOpenEnvV2),
                    ("faucet-open-v2", SawyerFaucetOpenEnvV2),
                    ("faucet-close-v2", SawyerFaucetCloseEnvV2),
                    ("hammer-v2", SawyerHammerEnvV2),
                    ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
                    ("handle-press-v2", SawyerHandlePressEnvV2),
                    ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
                    ("handle-pull-v2", SawyerHandlePullEnvV2),
                    ("lever-pull-v2", SawyerLeverPullEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
                    ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
                    ("reach-v2", SawyerReachEnvV2),
                    ("push-back-v2", SawyerPushBackEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    ("plate-slide-v2", SawyerPlateSlideEnvV2),
                    ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
                    ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
                    ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
                    ("soccer-v2", SawyerSoccerEnvV2),
                    ("stick-push-v2", SawyerStickPushEnvV2),
                    ("stick-pull-v2", SawyerStickPullEnvV2),
                    ("push-wall-v2", SawyerPushWallEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("reach-wall-v2", SawyerReachWallEnvV2),
                    ("reach-v2", SawyerReachEnvV2),
                    ("shelf-place-v2", SawyerShelfPlaceEnvV2),
                    ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
                    ("sweep-v2", SawyerSweepEnvV2),
                    ("window-open-v2", SawyerWindowOpenEnvV2),
                    ("window-close-v2", SawyerWindowCloseEnvV2),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("bin-picking-v2", SawyerBinPickingEnvV2),
                    ("box-close-v2", SawyerBoxCloseEnvV2),
                    ("hand-insert-v2", SawyerHandInsertEnvV2),
                    ("door-lock-v2", SawyerDoorLockEnvV2),
                    ("door-unlock-v2", SawyerDoorUnlockEnvV2),
                )
            ),
        ),
    )
)

ml45_train_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML45_V2["train"].items()
}

ml45_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML45_V2["test"].items()
}

ML45_ARGS_KWARGS = dict(
    train=ml45_train_args_kwargs,
    test=ml45_test_args_kwargs,
)


def create_hidden_goal_envs():
    hidden_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed=seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        hg_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = f"{env_name}-goal-hidden"
        hg_env_name = f"{hg_env_name}GoalHidden"
        HiddenGoalEnvCls = type(hg_env_name, (env_cls,), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None, render_mode=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()

            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.render_mode = render_mode
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")

        og_env_key = f"{env_name}-goal-observable"
        og_env_name = f"{og_env_name}GoalObservable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs()
ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()
