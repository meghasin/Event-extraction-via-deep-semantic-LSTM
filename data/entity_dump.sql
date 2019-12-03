select
em.id, ds.doc_id, e.type, em.context, em.stanford_sentence_id, em.stanford_token_start, em.stanford_token_end
from
entity e,
entity_mention em,
doc_set ds
where
ds.doc_id = e.doc_id and
e.id = em.entity_id and
ds.set_id = "training";
