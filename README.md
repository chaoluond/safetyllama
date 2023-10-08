# Introduction
The recent advance in Generative Pre-trained Transformer (GPT) technology (e.g. ChatGPT, LLaMA, Claude, Dolly, etc.) has been inspiring numerious tech companies and AI enthusiasts to utilize it to expore, experiment and deploy novel applications in search engine, recommendation system, work productivity tools, healthcare, advertising, and so on. 

GPT, or more broadly AI, is a double-edged sword. On one hand, it can tremendously boost the efficacy and efficiency of information retrieval and dissemination among humans. On the other hand, it might endager the safety and integrity of human beings if it is abused. In the US, the federal government has been enacting laws and policies to regulate AI ([reference1](https://www.whitehouse.gov/briefing-room/statements-releases/2023/07/21/fact-sheet-biden-harris-administration-secures-voluntary-commitments-from-leading-artificial-intelligence-companies-to-manage-the-risks-posed-by-ai/)). In Europe, an AI Act was proposed to regulate the development and use of AI ([reference2](https://www.europarl.europa.eu/news/en/headlines/society/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence))

To ensure the safety of AI chatbots, we introduce **Safety LLaMA** to monitor and detect unsafe responses from AI chatbots. Safety LLaMA is a 7B-chat LLaMA2 model, which was finetuned using the red team harmless dataset from Anthropics ([link](https://github.com/anthropics/hh-rlhf))  

# Methodology

