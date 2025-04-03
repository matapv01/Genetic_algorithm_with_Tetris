# **Genetic Algorithm with Tetris**  

This project applies a **Genetic Algorithm (GA)** to train an AI agent to play **Tetris** using a pre-built environment from the `Double-Agent-Tetris-main` repository. The goal is to evolve a high-performing AI agent capable of making optimal moves in the game.  

---

## **Project Overview**  

- The AI agent is trained using a **Genetic Algorithm**, where a population of agents evolves over multiple generations to improve performance.  
- The Tetris environment is provided in the **Double-Agent-Tetris-main** repository.  
- The trained agent can be tested using a provided **test script** or directly on **Google Colab** for ease of use.  

---

## **Getting Started**  

### **1. Clone the Repository**  
Ensure you have the required environment and dependencies installed before running the training or testing scripts.  

```bash
git clone https://github.com/matapv01/Genetic_algorithm_with_Tetris.git
cd Genetic_algorithm_with_Tetris
```


## **2. Train the AI Agent**

To train the AI agent using Genetic Algorithm, run the training script:

``` bash
python train.py
```

### **This script will**:

- Initialize a population of AI agents.
- Use a genetic algorithm to evolve agents over multiple generations.
- Select the best-performing agents based on a fitness function.
- Save the best agent for testing.

## **3. Test the AI Agent**


### **You can test the trained AI agent using the provided test script or via Google Colab.**

#### **Option 1**: Test Locally
- Upload the trained agent (agent.zip or agent1.zip & agent2.zip) and run:
```bash
python test_agent.py
```

#### **Option 2**: Test on Google Colab
- For convenience, you can run the test directly on Google Colab:
- Note:
  - If testing in a SingleENV environment, upload agent.zip (containing Agent.py).
  - If testing in a DoubleENV environment, upload both agent1.zip and agent2.zip (each containing Agent.py).

- Dependencies
  Ensure you have the required dependencies installed:
  ``` bash
  pip install -r requirements.txt
  ```


## Project Structure

📂 Double-Agent-Tetris-main  
│── 📂 agents/                # Folder containing trained agents  
│── 📂 environment/           # Tetris environment code  
│── 📜 train.py               # Script to train the AI agent  
│── 📜 test_agent.py          # Script to test the AI agent  
│── 📜 requirements.txt       # List of dependencies  
│── 📜 README.md              # Project documentation  



## License

This project is licensed under the MIT License - see the LICENSE file for details.
