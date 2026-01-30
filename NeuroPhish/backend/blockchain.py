import hashlib
import json
import time
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Blockchain:
    def __init__(self):
        self.chain: List[Dict[str, Any]] = []
        self.pending_threats: List[Dict[str, Any]] = []
        
        # Create the genesis block
        self.create_block(proof=1, previous_hash="0")

    def create_block(self, proof: int, previous_hash: str) -> Dict[str, Any]:
        """
        Create a new block in the blockchain.
        :param proof: The proof given by the Proof of Work algorithm
        :param previous_hash: Hash of previous block
        :return: New Block
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'threats': self.pending_threats,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        
        # Reset the current list of threats
        self.pending_threats = []
        
        self.chain.append(block)
        logger.info(f"🧱 New block mined! Index: {block['index']}")
        return block

    def get_last_block(self) -> Dict[str, Any]:
        return self.chain[-1]

    def add_threat(self, threat_data: Dict[str, Any]) -> int:
        """
        Add a new threat to the list of pending threats.
        :param threat_data: Dictionary containing threat details (url, type, timestamp, etc.)
        :return: The index of the Block that will hold this threat
        """
        self.pending_threats.append(threat_data)
        return self.get_last_block()['index'] + 1

    def proof_of_work(self, last_proof: int) -> int:
        """
        Simple Proof of Work Algorithm:
         - Find a number p' such that hash(pp') contains leading 4 zeroes
         - p is the previous proof, and p' is the new proof
        :param last_proof: <int>
        :return: <int>
        """
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof: int, proof: int) -> bool:
        """
        Validates the Proof: Does hash(last_proof, proof) contain 4 leading zeroes?
        :param last_proof: <int> Previous Proof
        :param proof: <int> Current Proof
        :return: <bool> True if correct, False if not.
        """
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    @staticmethod
    def hash(block: Dict[str, Any]) -> str:
        """
        Creates a SHA-256 hash of a Block
        :param block: <dict> Block
        :return: <str>
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def is_chain_valid(self, chain: List[Dict[str, Any]]) -> bool:
        """
        Determine if a given blockchain is valid
        :param chain: <list> A blockchain
        :return: <bool> True if valid, False if not
        """
        previous_block = chain[0]
        block_index = 1

        while block_index < len(chain):
            block = chain[block_index]
            
            # Check that the hash of the block is correct
            if block['previous_hash'] != self.hash(previous_block):
                return False

            # Check that the Proof of Work is correct
            previous_proof = previous_block['proof']
            proof = block['proof']
            if not self.valid_proof(previous_proof, proof):
                return False

            previous_block = block
            block_index += 1

        return True
