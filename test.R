if (!require('pacman', character.only = TRUE)) install.packages('pacman')
library(pacman)

packages <- c("tidyverse", "stringr", "tidytext", "ggthemes", "quanteda", "text2vec", "widyr", "irlba")

for (pkg in packages) {
  p_load(pkg, character.only = TRUE)
}

# install.packages("tidyverse")
# library(tidyverse)
# install.packages("stringr")
# library(stringr)

twts_sample <- readRDS(gzcon(url("https://github.com/cjbarrie/wordvectutorial/blob/main/data/wordembed/twts_corpus_sample.rds?raw=true")))

#create tweet id
twts_sample$postID <- row.names(twts_sample)

#create context window with length 6
tidy_skipgrams <- twts_sample %>%
  unnest_tokens(ngram, tweet, token = "ngrams", n = 6) %>%
  mutate(ngramID = row_number()) %>% 
  tidyr::unite(skipgramID, postID, ngramID) %>%
  unnest_tokens(word, ngram)

#calculate probabilities
skipgram_probs <- tidy_skipgrams %>%
  pairwise_count(word, skipgramID, diag = TRUE, sort = TRUE) %>% # diag = T means that we also count when the word appears twice within the window
  mutate(p = n / sum(n))


#calculate unigram probabilities (used to normalize skipgram probabilities later)
unigram_probs <- twts_sample %>%
  unnest_tokens(word, tweet) %>%
  count(word, sort = TRUE) %>%
  mutate(p = n / sum(n))

#normalize skipgram probabilities
normalized_prob <- skipgram_probs %>%
  filter(n > 20) %>% #filter out skipgrams with n <=20
  rename(word1 = item1, word2 = item2) %>%
  left_join(unigram_probs %>%
              select(word1 = word, p1 = p),
            by = "word1") %>%
  left_join(unigram_probs %>%
              select(word2 = word, p2 = p),
            by = "word2") %>%
  mutate(p_together = p / p1 / p2)



normalized_prob %>% 
  filter(word1 == "brexit") %>%
  arrange(-p_together) %>%
  select(word1, word2)





pmi_matrix <- normalized_prob %>%
  mutate(pmi = log10(p_together)) %>%
  cast_sparse(word1, word2, pmi)

#remove missing data
pmi_matrix@x[is.na(pmi_matrix@x)] <- 0

#run SVD
pmi_svd <- irlba(pmi_matrix, 256, maxit = 500)

#next we output the word vectors:
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)

dim(word_vectors)

nearest_words <- function(word_vectors, word){
  selected_vector = word_vectors[word,]
  mult = as.data.frame(word_vectors %*% selected_vector) #dot product of selected word vector and all word vectors
  
  mult %>%
    rownames_to_column() %>%
    rename(word = rowname,
           similarity = V1) %>%
    anti_join(get_stopwords(language = "en")) %>%
    arrange(-similarity)
  
}

brexit_synonyms <- nearest_words(word_vectors, "brexit")
brexit_synonyms


word_vectors[3,]


words <- c("Critical", "polite", "hostile", "decisive", "friendly", "diplomatic", "understanding", "philosophical", "able", "belligerent")


library(tm)
corpus <- Corpus(VectorSource(c("the cat sat on the mat", "the dog sat on the log")))
dtm <- DocumentTermMatrix(corpus)
co_occurrence_matrix <- crossprod(as.matrix(dtm))

# Example of the matrix
print(co_occurrence_matrix)
