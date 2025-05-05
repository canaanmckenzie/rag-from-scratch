# tests/test_generate_response.py

import unittest
from rag.generator import generate_response

class TestGeneratorModule(unittest.TestCase):
    def test_generate_answer_basic(self):
        query = "What is the poem about?"
        docs = [
            "Turning and turning in the widening gyre the falcon cannot hear the falconer...",
            "Things fall apart; the centre cannot hold; Mere anarchy is loosed upon the world..."
        ]
        response = generate_response(query, docs)
        
        # Print the response for debugging
        print("\nGenerated Response:\n", response)

        # Make sure the response isn't empty and includes some coherent reply
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 10)

if __name__ == '__main__':
    unittest.main()
