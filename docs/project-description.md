# Project Description

Analyzing the Impact of Multi-task Learning Strategies on Speech Task Performance

Multi-task learning (MTL) enables models to jointly learn multiple related tasks by sharing information across them, often leading to improved generalization. However, different MTL strategies vary significantly in how knowledge is shared between tasks, and their impact on individual task performance remains insufficiently understood. This project aims to systematically analyze how different MTL formulations influence performance across speech-related tasks. The study will use a speech LLM as the backbone to extract representations and support multi-task training, allowing consistent comparison across settings. To achieve this, a modular experimental framework will be developed to evaluate different MTL strategies on speech tasks. The study will implement and compare multiple approaches, including feature-based MTL and parameter-based MTL, using the same speech LLM setup. Using this framework, performance will be analyzed across content, speaker, and emotion tasks to understand how different sharing strategies influence each task. In addition, the effect of task imbalance will be examined, along with simple strategies to reduce its impact. Through this analysis, the project aims to provide a clearer understanding of how MTL design choices affect task performance and interaction

Key words: multi-task learning, machine learning, speech processing

Deliverables: The project will deliver a comprehensive Python library designed for MTL, integrating innovative methods for handling task imbalances. The project outcomes are expected to be published in the ACL2027.

Reference:
Sritharan, B., & Thayasivam, U. (2025, December). wavCSE: Learning Fixed-size Unified Speech Embeddings via Feature-based Multi-Task Learning. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (pp. 879-887).
Zhang, Y., & Yang, Q. (2021). A survey on multi-task learning. IEEE transactions on knowledge and data engineering, 34(12), 5586-5609.
Lin, B., & Zhang, Y. (2023). Libmtl: A Python Library for Deep Multitask Learning. Journal of Machine Learning Research, 24(1-7), 18.