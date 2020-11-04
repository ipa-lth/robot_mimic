from gym.envs.registration import register

register(id='RobotMimic-v0',
         entry_point='gym_robotmimic.envs:RobotMimicEnv',
)
