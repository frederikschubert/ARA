from typing import Tuple
import numpy as np
from rlpyt.envs.bullet.dynamic_locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.scenes.stadium import StadiumScene


class DynamicDynamicsBaseEnv(WalkerBaseBulletEnv):
    """
    Available Dynamics Parameters:
    - mass
    - lateralFriction
    - spinningFriction
    - rollingFriction
    - restitution
    - linearDamping
    - angularDamping
    - contactStiffness
    - contactDamping
    - frictionAnchor
    - activationState
    
    See https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.d6og8ua34um1
    and http://sdformat.org/spec?ver=1.8&elem=link
    """

    def __init__(self, changeDynamics_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.changeDynamics_kwargs = changeDynamics_kwargs
        self.original_dynamics = {}

    def change_dynamics(self):
        change_dynamics_kwargs = dict()
        for prop, value in self.changeDynamics_kwargs.items():
            if value["type"] == "range":
                lower, upper = value["limits"]
                new_val = np.random.uniform(lower, upper)
            elif value["type"] == "fixed":
                new_val = value["value"].value
            change_dynamics_kwargs[prop] = new_val
        if not self.original_dynamics:
            for name, body_part in self.parts.items():
                body_index = body_part.bodies[0]
                dynamics = self.get_dynamics(body_index, body_part.bodyPartIndex)
                self.original_dynamics[name] = dynamics
        for name, body_part in self.parts.items():
            body_index = body_part.bodies[0]
            changed_dynamics = dict()
            for k, v in change_dynamics_kwargs.items():
                if k == "mass" and not name == "torso":
                    continue
                if "Friction" in k and not name in self.robot.foot_list:
                    continue
                changed_dynamics[k] = self.original_dynamics[name][k] * v
            self._p.changeDynamics(
                body_index, body_part.bodyPartIndex, **changed_dynamics
            )

    def get_dynamics(self, body_index, body_part_index):
        (
            mass,
            lateral_friction,
            local_inertia_diagonal,
            local_inertial_pos,
            local_intertial_orn,
            restitution,
            rolling_friction,
            spinning_friction,
            contact_damping,
            contact_stiffness,
            body_type,
            collision_margin,
        ) = self._p.getDynamicsInfo(body_index, body_part_index)
        dynamics_info = dict(
            mass=mass,
            lateralFriction=lateral_friction,
            localInertiaDiagonal=local_inertia_diagonal,
            localInertialPos=local_inertial_pos,
            localIntertialOrn=local_intertial_orn,
            restitution=restitution,
            rollingFriction=rolling_friction,
            spinningFriction=spinning_friction,
            contactDamping=contact_damping,
            contactStiffness=contact_stiffness,
            bodyType=body_type,
            collisionMargin=collision_margin,
        )
        return dynamics_info

    def reset(self):
        r = super().reset()
        self.change_dynamics()
        return r
