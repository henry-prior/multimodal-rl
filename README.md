# Multimodal Reinforcement Learning

JAX implementations of the following multimodal reinforcement learning approaches.

- Dual-coding Episodic Memory from ["Grounded Language Learning Fast and Slow"](https://arxiv.org/pdf/2009.01719.pdf)

    The goal in this setting is for the agent to be presented with multiple objects with made up names following "This is a \_____" statements and to then carry out an instruction such as "Move the wazzle to the table." This task requires the agent to learn long-term language and vision representations for concepts like "This is a" and objects that carry over between episodes such as "table" while also being able to learn one-shot representations of novel objects and their names.

## Usage

Start by setting up the environment locally by running
```bash
poetry install
poetry shell
```

The learning environment depends on Docker and requires that the Docker Desktop program is running (on Mac). Once that's done you can run the default environment (fast mapping with 3 objects from the paper).

```bash
python fast_slow_learning/main.py
```
