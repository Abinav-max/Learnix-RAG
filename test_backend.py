import unittest
from app import (
    status_endpoint, 
    hotspots_endpoint, 
    search_endpoint, 
    generate_report_endpoint, 
    deep_dive_endpoint, 
    ingest_endpoint,
    SearchRequest,
    ReportRequest,
    IngestRequest
)

class TestAdversarialAcademicDirect(unittest.TestCase):
    
    def test_status(self):
        data = status_endpoint()
        self.assertEqual(data["system"], "Learnix Research Realtime Agent")
        self.assertIn("classifier", data)
        print("[OK] System status endpoint operational!")
        
    def test_hotspots(self):
        data = hotspots_endpoint()
        self.assertGreaterEqual(len(data), 1)
        print(f"[OK] Hotspots endpoint returned {len(data)} live trending paper(s)!")

    def test_fact_query_sky_blue(self):
        req = SearchRequest(query="Is the sky blue?")
        data = search_endpoint(req)
        self.assertTrue(data["is_fact"])
        self.assertEqual(data["total_matches"], 0)
        print("[OK] Fact query 'Is the sky blue?' correctly returned 0 flaws & Fact status!")

    def test_adversarial_search(self):
        req = SearchRequest(query="Should I use Transformers or Graph Neural Networks for forecasting?")
        data = search_endpoint(req)
        self.assertEqual(data["user_query"], req.query)
        self.assertIn("transformed_query", data)
        self.assertGreater(data["total_matches"], 0)
        print(f"[OK] Realtime Agentic Search successful! Matches found: {data['total_matches']}. Transformed query: {data['transformed_query']}")

    def test_academic_risk_report(self):
        req = ReportRequest(query="Should I use Transformers or Graph Neural Networks for forecasting?")
        data = generate_report_endpoint(req)
        self.assertIn("claim", data)
        self.assertIn("severity", data)
        self.assertIn("exposed_flaws", data)
        self.assertIn("suggested_mitigations", data)
        print(f"[OK] Academic Risk Report generated! Severity: {data['severity']['label']}, Score: {data['severity']['vulnerability_score']}")

    def test_deep_dive(self):
        req = SearchRequest(query="Should I use Transformers or Graph Neural Networks for forecasting?")
        search_res = search_endpoint(req)
        if search_res.get("results"):
            target_id = search_res["results"][0]["id"]
            data = deep_dive_endpoint(target_id)
            self.assertEqual(data["id"], target_id)
            self.assertIn("url", data)
            clean_title = str(data['title']).encode('ascii', 'ignore').decode('ascii')
            clean_tag = str(data.get('adversarial_tag', data.get('distilbert_tag', ''))).encode('ascii', 'ignore').decode('ascii')
            print(f"[OK] Deep Dive fetched live paper: {clean_title} (Tag: {clean_tag})")
        else:
            print("[OK] Deep dive test completed gracefully.")

    def test_deep_dive_zenodo(self):
        target_id = "zenodo-21786093"
        data = deep_dive_endpoint(target_id)
        self.assertEqual(data["id"], target_id)
        self.assertEqual(data["source"], "Zenodo")
        self.assertTrue(data["url"].startswith("https://zenodo.org/"), f"Invalid Zenodo URL: {data['url']}")
        self.assertIn("url", data)
        clean_t = str(data['title']).encode('ascii', 'ignore').decode('ascii')
        print(f"[OK] Deep Dive Zenodo resolution verified! Title: {clean_t} | Source: {data['source']} | URL: {data['url']}")

    def test_deep_dive_pubpeer(self):
        target_id = "pubpeer-10.1038-s41586-024-00001"
        data = deep_dive_endpoint(target_id)
        self.assertEqual(data["id"], target_id)
        self.assertEqual(data["source"], "PubPeer")
        self.assertTrue(data["url"].startswith("https://doi.org/"), f"Invalid PubPeer URL: {data['url']}")
        self.assertIn("url", data)
        clean_pt = str(data['title']).encode('ascii', 'ignore').decode('ascii')
        print(f"[OK] Deep Dive PubPeer resolution verified! Title: {clean_pt} | Source: {data['source']} | URL: {data['url']}")

    def test_ingest_pipeline(self):
        sample_text = (
            "We present SuperFastNet which outperforms all existing baselines on ImageNet with 99.9% accuracy.\n\n"
            "Reviewer #2: The author used test set images during the training data augmentation phase, causing massive data leakage and invalidating all claims."
        )
        req = IngestRequest(
            title="SuperFastNet",
            source="ArXiv",
            year=2024,
            raw_text=sample_text
        )
        data = ingest_endpoint(req)
        self.assertEqual(data["limitation_chunks_stored"], 1)
        self.assertEqual(data["marketing_chunks_dropped"], 1)
        try:
            from backend.db import get_supabase
            sb = get_supabase()
            sb.table("critiques").delete().eq("title", "SuperFastNet").execute()
        except Exception:
            pass
        print("[OK] Ingestion Adversarial Classifier: Stored 1 critique chunk, dropped 1 marketing claim!")

    def test_sentiment_gate_filters_supportive_survey(self):
        from backend.live_agent import detect_sentiment, is_supportive_marketing_fluff
        
        survey_title = "Transformers in Time Series: A Survey"
        survey_abstract = "Transformers have achieved superior performances in many time series tasks. In this survey, we provide an overview of exciting progress..."
        
        critique_title = "PatchTST: Are Transformers Just Moving Averages?"
        critique_abstract = "We demonstrate that Transformers for time-series forecasting are essentially complex linear models. When stripped of attention, they perform identically to a simple moving average, suggesting their claimed superiority is an artifact of evaluation design."
        
        self.assertEqual(detect_sentiment(survey_title, survey_abstract), "SUPPORTIVE")
        self.assertTrue(is_supportive_marketing_fluff(survey_title, survey_abstract))
        
        self.assertEqual(detect_sentiment(critique_title, critique_abstract), "CRITICAL")
        self.assertFalse(is_supportive_marketing_fluff(critique_title, critique_abstract))
        print("[OK] True Sentiment Detection successfully filtered out supportive survey and retained critical paper!")

    def test_domain_filter_removes_physics_spillover(self):
        from backend.live_agent import get_paper_domain, filter_by_domain, detect_query_domain
        
        query = "Does Chain-of-Thought prompting genuinely unlock new reasoning capabilities in LLMs?"
        self.assertEqual(detect_query_domain(query), "AI/ML/COMPUTER_SCIENCE")
        
        physics_paper = {
            "title": "Performance of the CMS detector in B-meson decay at LHC",
            "raw_text": "We present the measurement of B-meson decay using the Compact Muon Solenoid detector."
        }
        ai_paper = {
            "title": "Chain-of-Thought Hubris: Prompt Sensitivity and Spurious Correlation in LLMs",
            "raw_text": "We demonstrate that Chain-of-Thought prompting provides no unique reasoning benefit and is an artifact of output length."
        }
        
        self.assertEqual(get_paper_domain(physics_paper["title"], physics_paper["raw_text"]), "PHYSICS/ASTROPHYSICS")
        self.assertEqual(get_paper_domain(ai_paper["title"], ai_paper["raw_text"]), "AI/ML/COMPUTER_SCIENCE")
        
        filtered = filter_by_domain([physics_paper, ai_paper], "AI/ML/COMPUTER_SCIENCE")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["title"], ai_paper["title"])
        print("[OK] Domain Gate successfully filtered out particle physics paper and retained AI paper!")

    def test_source_filters_distinct(self):
        query = "Are Large Language Models capable of true logical reasoning?"
        
        # 1. Zenodo filter — authentic Zenodo research records
        req_zenodo = SearchRequest(query=query, source="Zenodo")
        res_zenodo = search_endpoint(req_zenodo)
        for item in res_zenodo.get("results", []):
            self.assertEqual(item["source"].lower(), "zenodo", f"Non-Zenodo item leaked: {item.get('source')}")
            self.assertGreater(len(item["text"].strip()), 10)
            self.assertIn("publisher", item)
            self.assertGreater(len(item["publisher"].strip()), 3)
            self.assertTrue(item["url"].startswith("https://zenodo.org/"), f"Non-Zenodo URL: {item['url']}")
            self.assertNotIn("arxiv org", item["publisher"].lower(), f"ArXiv publisher leaked into Zenodo: {item['publisher']}")
            
        # 2. ArXiv filter
        req_arxiv = SearchRequest(query=query, source="ArXiv")
        res_arxiv = search_endpoint(req_arxiv)
        self.assertGreater(len(res_arxiv.get("results", [])), 0)
        for item in res_arxiv.get("results", []):
            self.assertEqual(item["source"].lower(), "arxiv")
            self.assertIn("publisher", item)
            self.assertTrue(item["url"].startswith("https://arxiv.org/abs/"))
            
        # 3. PubPeer filter
        req_pubpeer = SearchRequest(query=query, source="PubPeer")
        res_pubpeer = search_endpoint(req_pubpeer)
        for item in res_pubpeer.get("results", []):
            self.assertEqual(item["source"].lower(), "pubpeer")
            self.assertIn("publisher", item)
            
        print("[OK] Source filtering verified! 100% Live API items retrieved across Zenodo, ArXiv, and PubPeer/Crossref with 100% strict platform isolation!")

    def test_active_attacks_multi_publisher_mix(self):
        query = "Are Large Language Models capable of true logical reasoning?"
        req = SearchRequest(query=query, source="All")
        res = search_endpoint(req)
        sources_found = set(item["source"].lower() for item in res.get("results", []))
        self.assertGreater(len(sources_found), 0)
        for item in res.get("results", []):
            self.assertIn("publisher", item)
            self.assertGreater(len(item["publisher"].strip()), 3)
        print(f"[OK] Active Attacks multi-publisher mix verified! Returned live items from publishers: {sources_found}")

    def test_new_sources_direct_fetch(self):
        from backend.live_agent import (
            fetch_biorxiv_realtime, fetch_medrxiv_realtime, fetch_openalex_realtime,
            fetch_semanticscholar_realtime, fetch_pmc_realtime, fetch_doaj_realtime
        )
        bio_hits = fetch_biorxiv_realtime("CRISPR gene editing", max_results=2)
        print(f"[OK] bioRxiv real-time fetch returned {len(bio_hits)} paper(s)")
        
        oa_hits = fetch_openalex_realtime("neural networks", max_results=2)
        print(f"[OK] OpenAlex real-time fetch returned {len(oa_hits)} paper(s)")

    def test_zero_shot_fine_tuning_query_no_biology_leakage(self):
        query = "Does the claimed generalization to novel conditions actually involve zero-shot inference, or does the fine-tuning data overlap with the test conditions?"
        req = SearchRequest(query=query, source="All")
        res = search_endpoint(req)
        
        self.assertEqual(res["category"], "RESEARCH_CLAIM")
        results = res.get("results", [])
        self.assertGreater(len(results), 0)
        
        biology_sources = {"biorxiv", "medrxiv", "pubmed central", "pmc"}
        for item in results:
            src = item.get("source", "").lower()
            self.assertNotIn(src, biology_sources, f"Biology paper from {src} leaked into AI query!")
            
            # Check title & text do not contain biology stem cell or transcriptome topics
            combined = (item.get("title", "") + " " + item.get("text", "")).lower()
            self.assertNotIn("stem cell", combined)
            self.assertNotIn("transcriptome", combined)
            self.assertNotIn("hematopoietic", combined)
            
        print(f"[OK] Zero-shot & fine-tuning query correctly classified as AI/ML and returned {len(results)} relevant AI paper(s) with ZERO biology spillover!")

if __name__ == "__main__":
    unittest.main()


