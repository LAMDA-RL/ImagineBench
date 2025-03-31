from envs import make, ENV_ID_LIST


if __name__ == '__main__':
    for env_id in ENV_ID_LIST:
        print(f"env_id: {env_id}")
        env = make(env_id)
        done = False
        while not done:
            obs = env.reset()
            print(obs)
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            print(next_obs)
            print(reward)
            print(done)
            print(info)

    print("done")
