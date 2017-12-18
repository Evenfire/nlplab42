import embeddings as em

emb = em.EmbeddingsDictionary(max_words=10)

#3.i
# nei = emb.w2neighbors('geek', 10)
# print(nei)

#3.ii
# +-
def find_comb_neighbors(embeddings, comb_words, comb_op, top_k=10):
    lst = [embeddings.emb[embeddings.dictionary[w]] for w in comb_words]
    comb = lst[0]
    for i, w in enumerate(lst[1:]):
        comb = comb + w if comb_op[i] == '+' else comb - w
    neigemb = embeddings.emb2neighbors(comb, top_k)
    words = [embeddings.words[ind] for ind in neigemb[1]]
    for i in range(len(words)):
        if words[i] not in comb_words:
            return words[i]
    return ""

# find_comb_neighbors(emb, ['Spain', 'Tokyo', 'France'], ['+', '-'])
res = find_comb_neighbors(emb, ['and', 'to', 'in'], ['+', '-'], 3)
print(len(res))