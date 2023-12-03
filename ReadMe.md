Curious agent applied to a Minecraft parkour environment.

This project explores the concept of curiosity-driven exploration in the context of reinforcement learning using a custom Minecraft parkour environment. The aim is to investigate the effectiveness of the Intrinsic Curiosity Module (ICM) compared to random exploration in achieving greater environment coverage and skill acquisition.

Usage:
To run the experiment, you must have a Minecraft account and connect to the Manacube parkour environment. Importantly, change the minecraft mouse settings of RAW-INPUT to off, otherwise the bot will not be able to control where you look. Run python main.py to start the bot. Make sure the right screen is being recorded.

The experiment will run for 25 episodes for a curious agent by default. After completion the results of the curiousity score over each episode is plotted and saved to curiosity_score.png.

File Structure:
-   main.py: Initialises an agent and begins the training process
-   models.py: Contains code of all the models relevant for the REINFORCE algorithm and the ICM (including feature encoder)
-   environment_interface.py: Contains code for controlling the agent and interacting with the minecraft environment.
-   reinforce.py: Runs the REINFORCE algorithm, collecting trajectories and trains the policy and ICM.
