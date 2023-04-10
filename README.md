# Levenshtein Distance Embedding

String similarity is a commonly encountered topic in bioinformatics, notably in DNA sequencing. Levenshtein being one of the edit distances used to consider it.

In one of my earlier projects "DNA Sequencing" I encountered the K Closest Neighbors problem under the edit distance, which turns out to be a very hard problem.
Thankfully, the exact same problem has been studied thoroughly for the Euclidean distance and many rather efficient algorithms exist to solve it.

For that reason, I attempted to embed the space of DNA sequences under the Levenshtein distance into the Euclidean space under the standard norm, while preserving distances as much as possible. 

In order to achieve that, I proposed a Recurrent Neural Network whose structure is described in the paper present in this repository, along with other results I found along the way.
