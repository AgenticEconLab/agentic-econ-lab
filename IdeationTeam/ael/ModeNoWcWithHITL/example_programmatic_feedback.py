# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Example: Using the Multi-Agent System with Programmatic Feedback
This demonstrates how to run the two-round process without interactive prompts.
"""

from typing import List
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import BaseModel, Field


class HumanFeedback(BaseModel):
    """Model for human feedback."""
    round_number: int = Field(description="Feedback round number")
    relevant_papers: List[str] = Field(description="Titles of relevant papers")
    irrelevant_papers: List[str] = Field(description="Titles of irrelevant papers")
    missing_topics: List[str] = Field(description="Topics that should be explored more")
    additional_keywords: List[str] = Field(description="Additional keywords to include")
    comments: str = Field(description="General comments and guidance")


def run_with_programmatic_feedback():
    """Run the multi-agent system with pre-defined feedback."""
    
    # Import after adding to path
    from importlib import import_module
    sourcing_module = import_module("1-SourcingStage")
    MultiAgentOrchestrator = sourcing_module.MultiAgentOrchestrator
    
    research_topic = "Agent-based modeling in macroeconomics and monetary policy"
    
    print("="*70)
    print("MULTI-AGENT LITERATURE SEARCH - PROGRAMMATIC FEEDBACK MODE")
    print("="*70)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # ROUND 1: Initial search
    print("\n[ROUND 1] Running initial search...")
    round1_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=1,
        feedback=None,
        max_results_per_agent=8
    )
    
    # Display Round 1 results
    orchestrator.print_summary(round_number=1, top_n=10)
    orchestrator.save_results("literature_results.csv", round_number=1)
    
    # Create programmatic feedback based on Round 1 results
    # In a real scenario, you would analyze round1_results and create feedback
    feedback = HumanFeedback(
        round_number=1,
        relevant_papers=[
            # Example: Papers that were found to be relevant
            "Agent-based models in macroeconomics",
            "Heterogeneous agent models"
        ],
        irrelevant_papers=[
            # Example: Papers that were off-topic
            "Pure microeconomic theory"
        ],
        missing_topics=[
            # Topics that need more coverage
            "central bank digital currencies",
            "climate change and monetary policy",
            "machine learning in macroeconomic forecasting"
        ],
        additional_keywords=[
            # New search terms to include
            "CBDC",
            "green monetary policy",
            "neural networks",
            "heterogeneous expectations",
            "bounded rationality"
        ],
        comments="Focus more on recent empirical applications and policy-relevant research. Include more work on digital currencies and climate-related monetary policy."
    )
    
    print("\n[FEEDBACK] Programmatic feedback created:")
    print(f"  - Relevant papers: {len(feedback.relevant_papers)}")
    print(f"  - Missing topics: {', '.join(feedback.missing_topics)}")
    print(f"  - Additional keywords: {', '.join(feedback.additional_keywords)}")
    print(f"  - Comments: {feedback.comments[:100]}...")
    
    # ROUND 2: Refined search with feedback
    print("\n[ROUND 2] Running refined search with feedback...")
    round2_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=2,
        feedback=feedback,
        max_results_per_agent=8
    )
    
    # Display Round 2 results
    orchestrator.print_summary(round_number=2, top_n=10)
    orchestrator.save_results("literature_results.csv", round_number=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Round 1: {len(round1_results)} papers")
    print(f"Round 2: {len(round2_results)} papers")
    print(f"Total unique papers: {len(set([p.title for p in orchestrator.all_results]))}")
    
    # Analyze improvement from feedback
    print(f"\n{'='*70}")
    print("FEEDBACK IMPACT ANALYSIS")
    print(f"{'='*70}")
    
    # Check if missing topics appeared in Round 2
    round2_titles_abstracts = " ".join([
        f"{p.title} {p.abstract}".lower() 
        for p in round2_results
    ])
    
    print("\nMissing topics coverage in Round 2:")
    for topic in feedback.missing_topics:
        if topic.lower() in round2_titles_abstracts:
            print(f"  ✓ {topic}: Found in results")
        else:
            print(f"  ✗ {topic}: Not found")
    
    print("\nAdditional keywords coverage in Round 2:")
    for keyword in feedback.additional_keywords:
        if keyword.lower() in round2_titles_abstracts:
            print(f"  ✓ {keyword}: Found in results")
        else:
            print(f"  ✗ {keyword}: Not found")
    
    # Save all results
    orchestrator.save_results("literature_results_all_rounds.csv")
    
    print(f"\n{'='*70}")
    print("PROCESS COMPLETE")
    print(f"{'='*70}")
    print("Files created:")
    print("  - round1_literature_results.csv")
    print("  - round2_literature_results.csv")
    print("  - literature_results_all_rounds.csv")


def run_with_custom_topic():
    """Example with a different research topic."""
    
    from importlib import import_module
    sourcing_module = import_module("1-SourcingStage")
    MultiAgentOrchestrator = sourcing_module.MultiAgentOrchestrator
    
    # Custom research topic
    research_topic = "Behavioral economics and nudge theory in public policy"
    
    orchestrator = MultiAgentOrchestrator()
    
    # Round 1
    round1_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=1,
        feedback=None,
        max_results_per_agent=6
    )
    
    orchestrator.print_summary(round_number=1, top_n=8)
    
    # Custom feedback
    feedback = HumanFeedback(
        round_number=1,
        relevant_papers=[],
        irrelevant_papers=[],
        missing_topics=[
            "choice architecture",
            "default options",
            "framing effects"
        ],
        additional_keywords=[
            "libertarian paternalism",
            "behavioral insights",
            "randomized controlled trials"
        ],
        comments="Focus on empirical studies with policy applications"
    )
    
    # Round 2
    round2_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=2,
        feedback=feedback,
        max_results_per_agent=6
    )
    
    orchestrator.print_summary(round_number=2, top_n=8)
    orchestrator.save_results(f"behavioral_econ_results.csv")


if __name__ == "__main__":
    print("\nChoose mode:")
    print("1. Run with programmatic feedback (default example)")
    print("2. Run with custom topic")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        run_with_programmatic_feedback()
    elif choice == "2":
        run_with_custom_topic()
    elif choice == "3":
        run_with_programmatic_feedback()
        print("\n" + "="*70)
        print("STARTING SECOND EXAMPLE")
        print("="*70 + "\n")
        run_with_custom_topic()
    else:
        print("Invalid choice. Running default example.")
        run_with_programmatic_feedback()
