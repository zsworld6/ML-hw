// #let theme_color = rgb("#652d89")
#let theme_color = gradient.linear(blue, black)
#let title_color = blue

#set text(font: "Noto Sans CJK SC", size: 34pt, fallback: false)
#set page(margin: 0em, width: 33.11in, height: 46.81in)
#set par(leading: 0.52em)
#let head_height = 4.48in
#let foot_height = 1.23in

#let header = [
  #set text(fill: white, weight: "bold")
  #show : rest => grid(
    columns: (1fr, 12em),
    rest,
    align(right + top, image("images/sjtu.png", width: 12em)),
  )
  #set text(size: 72pt)
  Using LLM to Generate Code for Classification and Regression Tasks \
  #set text(size: 40pt)
  Shiqi Zhang , 
  Shengyao Chen ,
  Pengrui Lu 

]

#let footer = [
  #set text(fill: white, weight: "bold")
  #set align(horizon)
  #grid(
    columns: (1fr, 1fr, 1fr),
    align: (left, center, right),
    [
      CS3308 Machine Learning, ACM Class
    ],
    [ /* #image("wkp-wechat.svg", width: 1em)*/ ],
    link("https://acm.sjtu.edu.cn"),
  )
]

#show : (rest) => {
  let head = block(
    fill: theme_color,
    width: 100%,
    height: head_height,
    header,
    spacing: 0em,
    inset: 2em,
  )
  let main = block(
    inset: (x: 2em, bottom: 1em, top: 2em),
    height: 100% - head_height - foot_height,
    spacing: 0em,
    fill: white,
    columns(2, gutter: 1em, rest),
  )
  let foot = block(
    fill: theme_color,
    width: 100%,
    height: foot_height,
    inset: (x: 1em),
    spacing: 0em,
    footer,
  )
  head
  main
  foot
}

// #set heading(
//   numbering: (..numbers) =>
//     if numbers.pos().len() <= 1 {
//       return numbering("1", ..numbers)
//     }
// )

#show heading.where(level: 1): set text(66pt, weight: "bold", fill: title_color)

// #show heading.where(level: 2): set text(48pt, weight: "bold", fill: theme_color)
#show heading: it => {
  v(-0.5em)
  it
}

= Background

== Reflexion

#image("images/reflexion.png", width: 50%)

== CAAFE

#image("images/CAAFE.png", width: 70%)

= Terminologies

== Reflexion
Iteratively analyze and refine LLM outputs through self-evaluation.
*generator*: LLM that generates code and self-Reflection.

*executor*: Executes the generated code and provides feedback.

== CAAFE
Provides context-sensitive feedback for automated feature engineering. Only used in classification tasks.

== Supervised Fine-Tuning
Trains LLMs on specific tasks using labeled datasets.

== LoRA
Efficient fine-tuning with low-rank matrices for large models.

== LLaMA-Factory
Framework for efficient fine-tuning and deployment of LLMs.

= Task

== Task Description
Generate code for machine learning tasks, focusing on classification and regression.


#grid(columns: (1fr, 1fr, 1fr),
align(horizon, image("images/Task.png",width: 100%)), align(horizon + center,
image("images/generating.png",width: 70%),
),
align(horizon, image("images/gen_code.png")))


= Challenges

#show heading.where(level: 2): it => text(it.body)

== Code Correctness 
: LLMs always generate wrong code.

== Insufficient GPUs
: Deploying or finetuning a large model requires too much computing power.

== Limited Data
: Few datasets are available for fine-tuning code generation, especially for classification/regression tasks.

= Our Approach

== Apply Reflexion
1. *Initial Generation*: Model generates code based on the input task.
2. *Evaluation*: The executor evaluates the generated code and provides feedback(error, performance, etc.).
3. *Self-Reflection Generation*: The generator model generates self-reflection words based on the code generated and the feedback.
4. *Code Refinement*: The generator model refines the code based on the self-reflection words.
5. *Iterative Process*: These steps are repeated until the generated code meets the 
desired quality standards or a preset number of iterations is reached.

== Apply CAAFE

*prompt engineering*: Designing prompts to guide the generator model to generate code using CAAFE.

== Supervised Fine-Tuning

*LoRA*: Low-Rank Adaptation (LoRA) fine-tunes large language models efficiently by introducing low-rank matrices, reducing parameters and computational overhead while maintaining or enhancing performance.

= Experiments

== Setup

=== Data Preparation
1. *Data Collection*: Datasets were collected from Kaggle, including both simple and complex feature datasets.
2. *Data Preprocessing*: Steps included extracting task descriptions, target labels, and several example rows from the datasets.

=== Datasets
- *Tasks*: House Price Dataset, Spaceship Titanic, Mobile Price Classification, etc.
- *Fine-tuning*: AlpacaCode

=== Compared Models
We compared the following models in our experiments:
- *Code Llama 7B*: A model designed for code generation tasks.
- *Qwen 0.5-1.5B Chat*: A chat-oriented model with varying parameter sizes.
- *Qwen 7B Chat*: A larger chat-oriented model.
- *Llama3 8B Instruct*: An instruction-tuned version of the Llama3 model.
- *Llama-3-8b-Instruct-bnb-4bit*: A quantized version of the Llama3 8B Instruct model for efficient fine-tuning.

== Results

=== Reflexion
// Llama3 model easily generated high quality code in simple datasets like Mobile Price Classification, but struggled with complex datasets like House Price Dataset. So we focus on House Price Dataset.
#figure(
  table(
    columns: 4,
    [model], [Dataset], [None], [Using Reflexion],
    [Llama3 8B Instruct], [House Price Dataset], [error or 0.19(rank 3800+)], [0.15(rank 2840)],
    [Qwen 7B Chat], [Spaceship Titanic], [cannot correctly generate code], [0.787(rank 1736)],
  ),
  caption: [performances of Reflexion],  
)

=== CAAFE
#figure(
  table(
    columns: 4,
    [model], [Dataset], [None], [Using CAAFE],    
    [Llama3 8B Instruct], [Spaceship Titanic], [0.75(rank 2150+)], [0.79(rank 1120)],
  ),
  caption: [performances of CAAFE],  
)

