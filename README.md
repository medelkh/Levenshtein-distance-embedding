# Levenshtein-distance-embedding
String similarity is a commonly encountered topic in bioinformatics, notably in DNA sequencing. Levenshtein being one of the edit distances used to consider it. In
my earlier project "DNA Sequencing" I encountered the K Closest Neighbors problem under the edit distance, which turns out to be a very hard problem, even under the
standard Euclidean distance, which thankfully has been studied thoroughly and many rather efficient algorithms exist to solve it. That's why I attempted to embed the
space of DNA sequences under the Levenshtein distance into the Euclidean space under the standard norm, while preserving distances as much as possible. In order to
achieve that, I went for a Recurrent Neural Network whose structure is described in the well detailed article present in this repository.
