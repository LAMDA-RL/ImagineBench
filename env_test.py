from rimaro import make, ENV_ID_LIST


if __name__ == '__main__':
    for env_id in ENV_ID_LIST:
        print(f"env_id: {env_id}")
        env = make(env_id)
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # print(obs)
            # print(next_obs)
            # print(reward)
            # print(done)
            # print(info)
        real_dataset, easy_dataset = env.get_dataset(level='easy')
        print(real_dataset)
        print(easy_dataset)

    print("done")
