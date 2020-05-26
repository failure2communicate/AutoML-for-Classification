# Model Name
QuestionAnswering_BideractionalAttentionFlow
 
# Input Type
json
 
# Input Description
A context, query pair. The query is a question that may be answered by text inside context. For example:
{"context": "A quick brown fox jumps over the lazy dog.", "query": "what color is the fox?"}
 
# Output Description
The text inside "context" that contains the answer to the query. For example:
"brown"
 
# Language
Python 3.6
 
# Description
This model is a neural network for finding the answer of a query based on a context.
The model was not built by UiPath but rather was built by the DS community, UiPath makes no guarantees on any performance metrics of the model.
The model is provided "as is" without warranty of any kind, either express or implied. Use at your own risk.   
     
    # Training data description (if available)
    The model was trained on SQuAD v1.1 data 
 
    # Benchmark (if available)
    The model achieved EM 68.1 over SQuAD v1.1 development data.   
     
    # Author(s) and Publication (if available)
    The model is based on a publication entitled "Bidirectional Attention Flow for Machine Comprehension" by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi.
    The publication can be found here:
    https://arxiv.org/pdf/1611.01603.pdf&nbsp;  
 
    # Implementatoin
    The model was packaged into a form consumable by AI Fabric. The pretrained model can be found here:
    https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow&nbsp;  
 
    # License
    MIT License
 
    Copyright (c) ONNX Project Contributors
 
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
 
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
  
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.