-- DELETE all UAE Embeddings from Embedding --
--DELETE FROM Embedding
--WHERE LOWER(model) LIKE '%uae%'
--    AND provider = 'together';
--

-- SELECT * FROM Embedding WHERE LOWER(model) LIKE "%uae%" AND embedding_vector = "[]";

SELECT DISTINCT provider, model FROM embedding;