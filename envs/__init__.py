class RIMAROEnv:
    pass


ENV_ID_LIST = [
    'Ball-v0',
    'MetaWorld-v0',
]

# 实现类似 gym 的 make 函数
def make(env_id: str, **kwargs) -> RIMAROEnv:
    if env_id not in ENV_ID_LIST:
        raise ValueError(f"env_id {env_id} is not supported.")
    
    env = None
    kwargs['level'] = kwargs.get('level', 'real')
    if env_id == 'Ball-v0':
        from envs.ball import BallEnv
        env = BallEnv(**kwargs)
    elif env_id == 'MetaWorld-v0':
        from envs.metaworld import MetaWorldEnv
        env = MetaWorldEnv(**kwargs)
    
    return env
