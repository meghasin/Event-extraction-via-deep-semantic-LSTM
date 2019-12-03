select
em.id, ds.doc_id, e.subtype, em.stanford_sentence_id, em.stanford_token
from
event_mention em,
event e,
doc_set ds
where
ds.doc_id = e.doc_id and
e.id = em.event_id and
ds.set_id = "dev";
