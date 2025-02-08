from langchain_ollama import ChatOllama


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


llm = ChatOllama(model="deepseek-r1:8b", temperature=0.7)

context = """
Benchmarking Multimodal Models for Fine-Grained
Image Analysis: A Comparative Study Across Diverse
Visual Features.
Evgenii Evstafev A
A University Information Services (UIS), University of Cambridge,
ABSTRACT
This article introduces a benchmark designed to evaluate the capabilities of multimodal models in analyzing and
interpreting images. The benchmark focuses on seven key visual aspects: main object, additional objects, background
detail, dominant colors, style, and viewpoint. A dataset of 14,580 images, generated from diverse text
prompts, was used to assess the performance of seven leading multimodal models. These models were evaluated
on their ability to accurately identify and describe each visual aspect, providing insights into their strengths and
weaknesses for comprehensive image understanding. The findings of this benchmark have significant implications
for the development and selection of multimodal models for various image analysis tasks.
TYPE OF PAPER AND KEYWORDS
Benchmarking, multimodal models, image analysis, computer vision, deep learning, image understanding, visual
features, model evaluation
1 INTRODUCTION
Multimodal models, capable of processing and integrating information from multiple modalities such as
text and images [1], have emerged as a powerful tool for
comprehensive image understanding [2]. These models
hold the potential to revolutionize various applications,
including image retrieval, content creation, and humancomputer interaction. However, evaluating their ability
to capture fine-grained details and contextual information remains a crucial challenge [3]. 
This article presents a benchmark for evaluating the performance of different multimodal models in identifying and analyzing
specific aspects of images, such as the main object, additional objects, background, details, dominant colors,
style, and viewpoint. By comparing their performance
across a range of tasks, this research aims to provide insights into the strengths and weaknesses of different
multimodal approaches for fine-grained image analysis.
2. BACKGROUND AND RELATED WORK
Multimodal models in computer vision use the interplay between different modalities, such as text and images, 
to achieve a more holistic understanding of visual
content [4]. This approach has shown promising results
in various tasks, including image captioning, visual
question answering, and image generation [5]. Recent
advancements in deep learning, particularly the development of transformer-based architectures, have further
accelerated progress in this field. Models like CLIP
(Contrastive Language-Image Pre-training [6]) have
demonstrated the ability to learn robust representations
that capture the semantic relationship between images
and text [7]. However, there is a need for standardized
benchmarks to evaluate the performance of multimodal
models in fine-grained image analysis, as existing
benchmarks often focus on broader tasks without explicitly assessing their ability to capture subtle details and
contextual information [8]. This research addresses this
gap by introducing a benchmark that specifically targets
the analysis of diverse visual features in images, enabling a more comprehensive evaluation of their capabilities.
3. METHODOLOGY
3.1 DATASET CREATION
The dataset creation process involved generating a
diverse set of image descriptions (prompts) by systematically
"""



class Author(BaseModel):
    name: str = Field(..., title="Author Name")
    affiliation: str = Field(..., title="Author Affiliation")

class Authors(BaseModel):
    authors: list[Author] = Field(..., title="Authors")


class ImagesUsed(BaseModel):
    number: int = Field(..., title="Number of Images Used")


parser = PydanticOutputParser(pydantic_object=ImagesUsed)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's query. Ensure you follow the format instructions: \n {format_instructions}"),
    ("human", "Question: {query}")
]).partial(format_instructions=parser.get_format_instructions())

chain = (
    {"query": RunnablePassthrough()}
    | prompt_template
    | llm
    | parser
)

result = chain.invoke(f"""
{context}

How many images were used in the dataset?
""")

print(result.number)

if result.number == 14580:
    print("Correct!")
























