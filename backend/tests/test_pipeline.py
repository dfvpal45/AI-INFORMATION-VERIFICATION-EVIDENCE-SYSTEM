import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

from app.pipeline import (
    TRUSTED_SOURCES,
    _confidence_from_evidence,
    _source_credibility,
    classify_claim,
    external_evidence,
    rank_evidence,
)


class ExternalEvidenceTests(unittest.TestCase):
    @patch('app.pipeline.httpx.Client')
    def test_search_results_are_returned_as_claim_specific_evidence(self, client_class):
        responses = {
            'https://news.google.com/rss/search': SimpleNamespace(
                content=(
                    b'<rss><channel><item><title>News claim update</title>'
                    b'<link>https://news.google.com/article</link>'
                    b'<description>News article snippet.</description>'
                    b'<source>Reuters</source></item></channel></rss>'
                ),
                raise_for_status=lambda: None,
            ),
            'https://html.duckduckgo.com/html/': SimpleNamespace(
                text=(
                    '<div class="result">'
                    '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ffact">'
                    'Relevant fact</a>'
                    '<a class="result__snippet">A snippet that matches the claim.</a>'
                    '</div>'
                ),
                raise_for_status=lambda: None,
            ),
        }

        def fake_get(url, **kwargs):
            return responses[url]

        client_class.return_value.__enter__.return_value.get.side_effect = fake_get

        evidence = external_evidence('A test claim')

        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]['title'], 'News claim update')
        self.assertEqual(evidence[0]['source'], 'Reuters')
        self.assertEqual(evidence[0]['credibility'], 0.94)
        self.assertEqual(evidence[1]['title'], 'Relevant fact')
        self.assertEqual(evidence[1]['text'], 'A snippet that matches the claim.')
        self.assertEqual(evidence[1]['url'], 'https://example.org/fact')

    def test_trusted_source_list_has_50_entries_and_scores_domains(self):
        self.assertEqual(len(TRUSTED_SOURCES), 50)
        self.assertEqual(_source_credibility('https://www.reuters.com/world'), 0.94)
        self.assertEqual(_source_credibility('https://en.wikipedia.org/wiki/Paris'), 0.72)
        self.assertEqual(_source_credibility('https://news.google.com/article', 'Reuters'), 0.94)
        self.assertEqual(_source_credibility('https://news.google.com/article', 'NASA (.gov)'), 0.97)
        self.assertEqual(_source_credibility('https://unknown.example/article'), 0.72)

    def test_unrelated_sources_are_not_returned_as_evidence(self):
        evidence = rank_evidence('The capital of France is Paris.', [
            {'title': 'Health topics', 'text': 'Public health guidance.', 'credibility': 0.96},
        ])

        self.assertEqual(evidence, [])

    def test_confidence_stays_within_requested_range(self):
        self.assertEqual(_confidence_from_evidence([]), 25.0)
        self.assertGreaterEqual(
            _confidence_from_evidence([
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'A'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'B'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'C'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'D'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'E'},
            ]),
            25.0,
        )
        self.assertLessEqual(
            _confidence_from_evidence([
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'A'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'B'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'C'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'D'},
                {'similarity': 0.9, 'rank_score': 0.9, 'credibility': 0.99, 'source': 'E'},
            ]),
            80.0,
        )

    @patch('app.pipeline.settings.enable_transformers', True)
    def test_nli_confidence_is_also_bounded(self):
        evidence = [{'text': 'A related factual source.', 'similarity': 0.4, 'credibility': 0.9}]
        fake_transformers = SimpleNamespace(pipeline=Mock())
        with patch.dict(sys.modules, {'transformers': fake_transformers}), patch.object(
            classify_claim, 'nli', return_value=[{'label': 'entailment', 'score': 0.99}], create=True
        ):
            verdict, confidence, _ = classify_claim('A factual claim.', evidence)
        self.assertEqual(verdict, 'Supported')
        self.assertEqual(confidence, 80.0)

    def test_analysis_explains_why_unverified(self):
        verdict, confidence, reason = classify_claim('A factual claim.', [
            {
                'title': 'A weakly related page',
                'text': 'Some vaguely related information.',
                'similarity': 0.12,
                'rank_score': 0.08,
                'credibility': 0.72,
                'source': 'example.org',
            },
        ])
        self.assertEqual(verdict, 'Unverified')
        self.assertGreaterEqual(confidence, 25.0)
        self.assertLessEqual(confidence, 80.0)
        self.assertIn('do not establish whether the claim is true or false', reason)
        self.assertIn('no independent high-credibility sources', reason)


if __name__ == '__main__':
    unittest.main()
