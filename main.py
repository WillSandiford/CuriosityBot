from reinforce import Agent

def main():
    agent = Agent()
    agent.train(2, 2, curious=True)

if __name__ == '__main__':
    main()
