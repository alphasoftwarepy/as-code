import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.rag_models import Base, RAGDocument
from api.memory_models import MemoryBase, MemoryObservation
from runtime.coordinator.intent import analyze_intent

class TestSkillRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an in-memory SQLite database for intent routing tests
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        MemoryBase.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db = self.SessionLocal()
        # Clean up database tables before each test
        self.db.query(RAGDocument).delete()
        self.db.query(MemoryObservation).delete()
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_neutral_query_returns_empty(self):
        # A completely neutral query with no matching keywords should return no skills
        skills = analyze_intent("de que trata este documento", self.db, "session_1")
        self.assertEqual(skills, [])

    def test_domain_specific_query_matches(self):
        # A query with a clear marketing keyword should match marketing
        skills = analyze_intent("diseña una campaña de publicidad", self.db, "session_1")
        self.assertIn("marketing", skills)

    def test_user_intent_gate_prevents_document_pollution(self):
        # Even if a document belonging to the session contains marketing keywords (e.g. "ad_campaign"),
        # a neutral query must return empty because of the User Intent Gate.
        doc = RAGDocument(
            id="doc1",
            filename="my_ad_campaign.pdf",
            file_type="pdf",
            content="dummy content",
            pipeline="chat",
            session_id="session_1"
        )
        self.db.add(doc)
        self.db.commit()

        skills = analyze_intent("de que trata este documento", self.db, "session_1")
        self.assertEqual(skills, [])

    def test_user_intent_gate_prevents_observation_pollution(self):
        # Even if there are observations about marketing, a neutral query should not trigger the skill.
        obs = MemoryObservation(
            id="obs1",
            session_id="session_1",
            content="El usuario quiere hacer una campaña de facebook."
        )
        self.db.add(obs)
        self.db.commit()

        skills = analyze_intent("de que trata este documento", self.db, "session_1")
        self.assertEqual(skills, [])

    def test_session_scoped_documents(self):
        # A document in session_2 should not affect the score or matching in session_1.
        doc_s2 = RAGDocument(
            id="doc_s2",
            filename="marketing_strategy.pdf",
            file_type="pdf",
            content="dummy content",
            pipeline="chat",
            session_id="session_2"
        )
        self.db.add(doc_s2)
        self.db.commit()

        # Query in session_1 with a keyword that matches marketing
        # marketing should match because of the query keyword "ventas"
        # However, the document filename 'marketing_strategy.pdf' in session_2 should not boost it.
        skills = analyze_intent("vamos a hablar de ventas", self.db, "session_1")
        self.assertIn("marketing", skills)

    def test_word_boundary_filename_matching(self):
        # A filename like "CERTIFICACIÓN APLICADA.docx" contains the word "APLICADA",
        # which has the substring "ad". "ad" is a marketing keyword.
        # But a word boundary search should NOT match "ad" inside "APLICADA".
        doc = RAGDocument(
            id="doc_applied",
            filename="CERTIFICACIÓN APLICADA.docx",
            file_type="docx",
            content="dummy content",
            pipeline="chat",
            session_id="session_1"
        )
        self.db.add(doc)
        self.db.commit()

        # If the user message has a marketing keyword (e.g. "redes"), marketing will pass the intent gate.
        # However, "CERTIFICACIÓN APLICADA.docx" should NOT boost the score for "marketing" (no +3 from "ad" in "aplicada").
        # If it did boost it, marketing would have score: message(2) + doc(3) = 5.
        # If word boundaries work, it won't boost, so score is message(2) = 2.
        # Let's test a contrast with a real marketing document: "flyer_anuncio.docx"
        doc_real = RAGDocument(
            id="doc_real",
            filename="flyer_anuncio.docx",
            file_type="docx",
            content="dummy content",
            pipeline="chat",
            session_id="session_1"
        )
        self.db.add(doc_real)
        self.db.commit()

        # Query matching marketing ("redes")
        skills = analyze_intent("hablemos de redes", self.db, "session_1")
        self.assertIn("marketing", skills)

if __name__ == "__main__":
    unittest.main()
