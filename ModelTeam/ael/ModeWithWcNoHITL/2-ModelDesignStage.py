# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Model Design Stage - Automated Mode (No Firecrawl, No HITL)
This script formulates formal mathematical models based on theoretical frameworks from Stage 1.

Pipeline:
1. ModelDesigner: Formal mathematical model formulation (complete equation system, constraints, dynamics)

Input: Theoretical frameworks from Stage 1 (theory_output.json)
Output: Formal mathematical model with complete equation system
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
# Add parent directories to path for shared imports
import sys
from pathlib import Path as _Path
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.json_repair import repair_json
from shared.parallel import parallel_map
from shared.auto_input import auto_input, get_default
from ModelTeam.ael.schemas.stage_outputs import (
    TheoreticalFramework, ModelVariable, ModelParameter, ModelEquation,
    ModelConstraint, ModelDynamics, EquilibriumDefinition, SolutionMethod,
    FormalMathematicalModel, ModelDesignOutput,
)

# Load environment variables
load_dotenv()


# ========== AGENT ==========

class ModelDesigner:
    """Agent for formulating formal mathematical models from theoretical frameworks."""
    
    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "ModelDesigner"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name,
        )
    
    def design_formal_model(
        self,
        framework: TheoreticalFramework,
        revision_request: Optional[Dict] = None
    ) -> FormalMathematicalModel:
        """Design a formal mathematical model from a theoretical framework.

        ``revision_request`` (a Model Revision Request from the data-feasibility loop —
        README.md, "Running") is optional and additive: when None
        the design is unchanged; when present, a revision directive is injected into
        variable definition so the model drops/relaxes the flagged (data-infeasible)
        variable(s).
        """

        print(f"\n[{self.agent_name}] Designing formal model for:")
        print(f"  {framework.framework_title}")

        # Optional data-feasibility revision directive (empty -> no behavior change).
        try:
            from DataTeam.ael.feasibility.adapters import format_revision_directive
            revision_note = format_revision_directive(revision_request)
        except Exception:
            revision_note = ""
        if revision_note:
            print("  [revision] applying data-feasibility MRR to variable definition")

        # Step 1: Define variables
        print(f"  Step 1: Defining model variables...")
        variables = self._define_variables(framework, revision_note)
        
        # Step 2: Define parameters
        print(f"  Step 2: Defining model parameters...")
        parameters = self._define_parameters(framework)
        
        # Step 3: Formulate complete equation system
        print(f"  Step 3: Formulating complete equation system...")
        equations = self._formulate_equations(framework, variables, parameters, revision_note)
        
        # Step 4: Specify constraints
        print(f"  Step 4: Specifying constraints...")
        constraints = self._specify_constraints(framework, variables, parameters)
        
        # Step 5: Define dynamics
        print(f"  Step 5: Defining model dynamics...")
        dynamics = self._define_dynamics(framework, variables, equations)
        
        # Step 6: Define equilibrium
        print(f"  Step 6: Defining equilibrium concept...")
        equilibrium = self._define_equilibrium(framework, equations)
        
        # Step 7: Specify solution method
        print(f"  Step 7: Specifying solution method...")
        solution_method = self._specify_solution_method(framework, equations, dynamics)
        
        # Step 8: Synthesize complete model
        print(f"  Step 8: Synthesizing complete formal model...")
        formal_model = self._synthesize_model(
            framework,
            variables,
            parameters,
            equations,
            constraints,
            dynamics,
            equilibrium,
            solution_method
        )
        
        print(f"[{self.agent_name}] Formal model designed: {formal_model.model_title}")
        return formal_model
    
    def _define_variables(
        self,
        framework: TheoreticalFramework,
        revision_note: str = ""
    ) -> List[ModelVariable]:
        """Define all model variables.

        ``revision_note`` (empty by default) prepends a data-feasibility revision
        directive; when empty the prompt is byte-identical to the unrevised run.
        """

        # Prepare context
        components_text = "\n".join([
            f"- {getattr(c, 'component_name', 'Unknown')}: {getattr(c, 'description', '')[:100]}"
            for c in framework.conceptual_components
        ])

        equations_text = "\n".join([
            f"- {getattr(e, 'equation_name', 'Unknown')}: {', '.join(getattr(e, 'variables', []))}"
            for e in framework.mathematical_formulations
        ])

        directive = (revision_note + "\n\n") if revision_note else ""

        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at defining variables for formal economic models."},
                {"role": "user", "content": """{directive}Define 10-20 key variables for this formal mathematical model.

Framework: {framework_title}
Theoretical Approach: {approach}

Components:
{components}

Existing Equations:
{equations}

For each variable, provide:
- variable_id: Short ID (e.g., "V1", "V2")
- variable_symbol: Math symbol (e.g., "c_t", "K", "pi")
- variable_name: Full name
- variable_type: endogenous, exogenous, state, control, or parameter
- description: Clear description
- domain: Domain/range (e.g., "R+", "[0,1]")
- time_subscript: true/false
- agent_subscript: If applicable (e.g., "i", "h")

Return as JSON with "variables" array.
Example: {{
  "variables": [
    {{
      "variable_id": "V1",
      "variable_symbol": "c_t",
      "variable_name": "Consumption",
      "variable_type": "control",
      "description": "Household consumption at time t",
      "domain": "R+",
      "time_subscript": true,
      "agent_subscript": ""
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    directive=directive,
                    framework_title=framework.framework_title,
                    approach=framework.theoretical_approach,
                    components=components_text,
                    equations=equations_text
                )}
            ])

            data = json.loads(repair_json(result))
            variables = [ModelVariable(**v) for v in data.get("variables", [])]
            
            return variables
        
        except Exception as e:
            print(f"    Error defining variables: {e}")
            return []
    
    def _define_parameters(
        self,
        framework: TheoreticalFramework
    ) -> List[ModelParameter]:
        """Define all model parameters."""
        
        equations_text = "\n".join([
            f"- {getattr(e, 'equation_name', 'Unknown')}: {', '.join(getattr(e, 'parameters', []))}"
            for e in framework.mathematical_formulations
        ])
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at defining parameters for formal economic models."},
                {"role": "user", "content": """Define 8-15 key parameters for this formal mathematical model.

Framework: {framework_title}
Theoretical Approach: {approach}

Existing Equations:
{equations}

For each parameter, provide:
- parameter_id: Short ID (e.g., "P1", "P2")
- parameter_symbol: Math symbol (e.g., "beta", "sigma", "alpha")
- parameter_name: Full name
- description: Economic interpretation
- typical_range: Typical range in literature
- calibration_source: How to calibrate/estimate

Return as JSON with "parameters" array.
Example: {{
  "parameters": [
    {{
      "parameter_id": "P1",
      "parameter_symbol": "beta",
      "parameter_name": "Discount factor",
      "description": "Household's discount factor for future utility",
      "typical_range": "[0.95, 0.99]",
      "calibration_source": "Standard macro literature, typically 0.96-0.99"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    framework_title=framework.framework_title,
                    approach=framework.theoretical_approach,
                    equations=equations_text
                )}
            ])

            data = json.loads(repair_json(result))
            # Per-item construction: a single malformed parameter dict must NOT
            # discard the whole list (the old list-comp raised on the first bad
            # item -> a model with 0 parameters). ModelParameter now inherits
            # LLMCoercedModel so most shape mismatches auto-coerce; anything still
            # invalid is skipped while the rest are kept.
            parameters = []
            for p in data.get("parameters", []):
                try:
                    parameters.append(ModelParameter(**p))
                except Exception as item_err:
                    print(f"    Skipping invalid parameter {p.get('parameter_symbol', '?') if isinstance(p, dict) else p!r}: {item_err}")
                    continue

            return parameters

        except Exception as e:
            print(f"    Error defining parameters: {e}")
            return []
    
    def _formulate_equations(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        parameters: List[ModelParameter],
        revision_note: str = "",
    ) -> List[ModelEquation]:
        """Formulate complete equation system.

        ``revision_note`` carries the data-feasibility revision directive: the framework's
        existing equations still name the removed variables, so the equation step must see the
        directive too or it reintroduces them."""
        directive = ""
        if revision_note:
            directive = (revision_note + "\nUse only the variables listed below; an existing "
                         "equation that needs a removed variable must be rewritten or dropped.\n\n")
        
        vars_text = "\n".join([f"- {v.variable_symbol}: {v.variable_name}" for v in variables[:15]])
        params_text = "\n".join([f"- {p.parameter_symbol}: {p.parameter_name}" for p in parameters[:10]])
        
        existing_eqs = "\n".join([
            f"- {getattr(e, 'equation_name', 'Unknown')}: {getattr(e, 'equation_latex', '')}"
            for e in framework.mathematical_formulations
        ])
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at formulating complete equation systems for economic models."},
                {"role": "user", "content": """{directive}Formulate a complete equation system (12-20 equations) for this formal model.

Framework: {framework_title}
Approach: {approach}
Model Structure: {structure}

Variables:
{variables}

Parameters:
{parameters}

Existing Equations:
{existing_equations}

For each equation, provide:
- equation_id: Short ID (e.g., "EQ1", "EQ2")
- equation_type: behavioral, equilibrium, identity, constraint, or law_of_motion
- equation_name: Name
- equation_latex: LaTeX (use \\\\frac, \\\\sum, etc.)
- equation_plain: Plain text version
- variables_used: List of variable symbols
- parameters_used: List of parameter symbols
- interpretation: Economic interpretation
- derivation: Brief derivation notes
- timing: Timing convention if relevant

Include: utility/profit maximization FOCs, budget constraints, market clearing, laws of motion, identities.

Return as JSON with "equations" array.
Example: {{
  "equations": [
    {{
      "equation_id": "EQ1",
      "equation_type": "behavioral",
      "equation_name": "Euler Equation",
      "equation_latex": "u'(c_t) = \\\\beta E_t[u'(c_{{t+1}})(1+r_{{t+1}})]",
      "equation_plain": "u'(c_t) = beta * E_t[u'(c_{{t+1}}) * (1 + r_{{t+1}})]",
      "variables_used": ["c_t", "c_{{t+1}}", "r_{{t+1}}"],
      "parameters_used": ["beta"],
      "interpretation": "Intertemporal consumption optimality",
      "derivation": "FOC from household utility maximization",
      "timing": "Period t decision"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    framework_title=framework.framework_title,
                    approach=framework.theoretical_approach,
                    structure=framework.model_structure[:300],
                    variables=vars_text,
                    parameters=params_text,
                    existing_equations=existing_eqs,
                    directive=directive,
                )}
            ], max_tokens=6000)  # bound the equation system (12-20 eqs fit ~5k tok); an 8192 cap
            # lets this call run close to AEL_LLM_TIMEOUT=600s. This call is on the critical
            # path of the parallelized per-model loop.

            data = json.loads(repair_json(result))
            equations = [ModelEquation(**e) for e in data.get("equations", [])]

            return self._enforce_symbolic_equations(equations)

        except Exception as e:
            print(f"    Error formulating equations: {e}")
            return []

    @staticmethod
    def _n_symbolic(equations) -> int:
        return sum(1 for e in equations if "=" in (getattr(e, "equation_plain", "") or ""))

    def _enforce_symbolic_equations(self, equations):
        """A model may emit its equations as PROSE sentences
        ("Marginal utility of consumption today equals...") — downstream parsers refuse
        honestly and the model becomes ungenerable/uncalibratable. Deterministic format
        gate: if MOST equations lack '=', ONE re-prompt demanding symbolic form; keep
        whichever set has more symbolic equations; still prose -> proceed disclosed
        (the parsers remain the judges — this gate only enforces the format contract)."""
        if not equations or self._n_symbolic(equations) * 2 >= len(equations):
            return equations
        n_prose = len(equations) - self._n_symbolic(equations)
        print(f"    [symbolic-gate] {n_prose}/{len(equations)} equations are prose (no '=') — "
              f"one symbolic-form re-prompt")
        try:
            retry = self.llm.invoke([
                {"role": "system", "content": "You are an expert at formulating complete "
                                              "equation systems for economic models."},
                {"role": "user", "content":
                    "Rewrite these model equations in STRICT SYMBOLIC FORM. Every "
                    "equation_plain MUST be an algebraic statement containing '=' built "
                    "from the variable/parameter SYMBOLS (e.g. 'c + a_next = (1+r)*a + "
                    "w*l'), NEVER an English sentence. Keep every equation_id, name and "
                    "interpretation unchanged. Return the same JSON shape "
                    '({"equations": [...]}), ONLY the JSON object.\n\n'
                    + json.dumps({"equations": [e.model_dump() for e in equations]},
                                 ensure_ascii=False, default=str)[:8000]},
            ], max_tokens=6000)
            redata = json.loads(repair_json(retry))
            re_eqs = [ModelEquation(**e) for e in redata.get("equations", [])]
            if re_eqs and self._n_symbolic(re_eqs) > self._n_symbolic(equations):
                print(f"    [symbolic-gate] re-prompt: {self._n_symbolic(re_eqs)}/{len(re_eqs)} "
                      f"now symbolic")
                return re_eqs
            print("    [symbolic-gate] re-prompt did not improve — proceeding with the original "
                  "set, disclosed by downstream refusals")
        except Exception as e:
            print(f"    [symbolic-gate] re-prompt failed ({e}) — keeping original equations")
        return equations
    
    def _specify_constraints(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        parameters: List[ModelParameter]
    ) -> List[ModelConstraint]:
        """Specify model constraints."""
        
        vars_text = "\n".join([f"- {v.variable_symbol}: {v.variable_name}" for v in variables[:15]])
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at specifying constraints for economic models."},
                {"role": "user", "content": """Specify 5-10 key constraints for this formal model.

Framework: {framework_title}

Variables:
{variables}

For each constraint, provide:
- constraint_id: Short ID (e.g., "C1", "C2")
- constraint_type: budget, resource, feasibility, non_negativity, or boundary
- constraint_latex: LaTeX representation
- constraint_plain: Plain text
- description: Description
- binding_conditions: When it binds

Return as JSON with "constraints" array.
Example: {{
  "constraints": [
    {{
      "constraint_id": "C1",
      "constraint_type": "budget",
      "constraint_latex": "c_t + s_t \\\\leq w_t + (1+r_t)s_{{t-1}}",
      "constraint_plain": "c_t + s_t <= w_t + (1+r_t)*s_{{t-1}}",
      "description": "Household budget constraint",
      "binding_conditions": "Always binding in optimal solution"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    framework_title=framework.framework_title,
                    variables=vars_text
                )}
            ])

            data = json.loads(repair_json(result))
            constraints = [ModelConstraint(**c) for c in data.get("constraints", [])]
            
            return constraints
        
        except Exception as e:
            print(f"    Error specifying constraints: {e}")
            return []
    
    def _define_dynamics(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        equations: List[ModelEquation]
    ) -> ModelDynamics:
        """Define model dynamics."""
        
        state_vars = [v.variable_symbol for v in variables if v.variable_type == "state"]
        control_vars = [v.variable_symbol for v in variables if v.variable_type == "control"]
        transition_eqs = [e.equation_id for e in equations if e.equation_type == "law_of_motion"]
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at defining dynamic structures for economic models."},
                {"role": "user", "content": """Define the dynamic structure for this model.

Framework: {framework_title}
Approach: {approach}

State Variables: {state_vars}
Control Variables: {control_vars}
Transition Equations: {transition_eqs}

Provide:
- time_structure: "Discrete" or "Continuous"
- time_horizon: "Finite" or "Infinite"
- state_variables: List of state variable symbols
- control_variables: List of control variable symbols
- transition_equations: List of transition equation IDs
- initial_conditions: List of initial conditions
- terminal_conditions: List of terminal conditions (if finite horizon)

Return as JSON.
Example: {{
  "time_structure": "Discrete",
  "time_horizon": "Infinite",
  "state_variables": ["k_t", "z_t"],
  "control_variables": ["c_t", "l_t"],
  "transition_equations": ["EQ5", "EQ6"],
  "initial_conditions": ["k_0 given", "z_0 ~ N(0, sigma_z^2)"],
  "terminal_conditions": []
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    framework_title=framework.framework_title,
                    approach=framework.theoretical_approach,
                    state_vars=", ".join(state_vars) if state_vars else "None specified",
                    control_vars=", ".join(control_vars) if control_vars else "None specified",
                    transition_eqs=", ".join(transition_eqs) if transition_eqs else "None specified"
                )}
            ])

            data = json.loads(repair_json(result))
            dynamics = ModelDynamics(**data)
            
            return dynamics
        
        except Exception as e:
            print(f"    Error defining dynamics: {e}")
            return ModelDynamics(
                time_structure="Discrete",
                time_horizon="Infinite",
                state_variables=state_vars,
                control_variables=control_vars,
                transition_equations=transition_eqs,
                initial_conditions=[],
                terminal_conditions=[]
            )
    
    def _define_equilibrium(
        self,
        framework: TheoreticalFramework,
        equations: List[ModelEquation]
    ) -> EquilibriumDefinition:
        """Define equilibrium concept."""
        
        eq_eqs = [e.equation_id for e in equations if e.equation_type == "equilibrium"]
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at defining equilibrium concepts for economic models."},
                {"role": "user", "content": """Define the equilibrium concept for this model.

Framework: {framework_title}
Equilibrium Concept: {eq_concept}

Provide:
- equilibrium_type: Type (e.g., "Competitive Equilibrium", "Nash Equilibrium", "Rational Expectations")
- equilibrium_conditions: List of conditions (3-5)
- market_clearing: Market clearing conditions (2-4)
- optimality_conditions: Optimality conditions (3-5)
- consistency_requirements: Consistency requirements (2-3)

Return as JSON.
Example: {{
  "equilibrium_type": "Competitive Equilibrium with Rational Expectations",
  "equilibrium_conditions": ["Households optimize", "Firms optimize", "Markets clear"],
  "market_clearing": ["Labor market: L_d = L_s", "Goods market: Y = C + I"],
  "optimality_conditions": ["Euler equation holds", "Labor supply FOC holds"],
  "consistency_requirements": ["Expectations are rational", "Budget constraints satisfied"]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    framework_title=framework.framework_title,
                    eq_concept=framework.equilibrium_concept or "Not specified"
                )}
            ])

            data = json.loads(repair_json(result))
            equilibrium = EquilibriumDefinition(**data)
            
            return equilibrium
        
        except Exception as e:
            print(f"    Error defining equilibrium: {e}")
            return EquilibriumDefinition(
                equilibrium_type="General Equilibrium",
                equilibrium_conditions=[],
                market_clearing=[],
                optimality_conditions=[],
                consistency_requirements=[]
            )
    
    def _specify_solution_method(
        self,
        framework: TheoreticalFramework,
        equations: List[ModelEquation],
        dynamics: ModelDynamics
    ) -> SolutionMethod:
        """Specify solution methodology."""
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at specifying solution methods for economic models."},
                {"role": "user", "content": """Specify the solution methodology for this model.

Framework: {framework_title}
Solution Approach: {solution_approach}
Time Structure: {time_structure}
Number of Equations: {num_equations}

Provide:
- solution_approach: "Analytical", "Numerical", or "Simulation"
- solution_steps: Ordered steps to solve (5-8 steps)
- computational_methods: Methods needed (e.g., "Linearization", "Value function iteration")
- software_requirements: Software/packages (e.g., "MATLAB", "Dynare", "Python (NumPy, SciPy)")
- expected_challenges: Computational challenges (2-4)

Return as JSON.
Example: {{
  "solution_approach": "Numerical",
  "solution_steps": ["Step 1: Calibrate parameters", "Step 2: Linearize around steady state", "Step 3: Solve linear system"],
  "computational_methods": ["Log-linearization", "Blanchard-Kahn method"],
  "software_requirements": ["Dynare", "MATLAB"],
  "expected_challenges": ["High dimensionality", "Curse of dimensionality"]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    framework_title=framework.framework_title,
                    solution_approach=framework.solution_approach,
                    time_structure=dynamics.time_structure,
                    num_equations=len(equations)
                )}
            ])

            data = json.loads(repair_json(result))
            solution_method = SolutionMethod(**data)
            
            return solution_method
        
        except Exception as e:
            print(f"    Error specifying solution method: {e}")
            return SolutionMethod(
                solution_approach="Numerical",
                solution_steps=[],
                computational_methods=[],
                software_requirements=[],
                expected_challenges=[]
            )
    
    def _synthesize_model(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        parameters: List[ModelParameter],
        equations: List[ModelEquation],
        constraints: List[ModelConstraint],
        dynamics: ModelDynamics,
        equilibrium: EquilibriumDefinition,
        solution_method: SolutionMethod
    ) -> FormalMathematicalModel:
        """Synthesize complete formal model."""
        
        model_title = f"Formal Model: {framework.framework_title}"
        
        # Generate model summary
        model_summary = f"""
This formal mathematical model operationalizes the theoretical framework '{framework.framework_title}'.
It consists of {len(variables)} variables, {len(parameters)} parameters, {len(equations)} equations, and {len(constraints)} constraints.
The model uses a {dynamics.time_structure.lower()} time structure with {dynamics.time_horizon.lower()} horizon.
Equilibrium is defined as {equilibrium.equilibrium_type}.
        """.strip()
        
        # Notation conventions
        notation = """
Time subscripts: t, t+1, t-1
Agent subscripts: i, h, f (if applicable)
Expectations: E_t[·]
Derivatives: ∂/∂x or x'
        """.strip()
        
        # Recap assumptions
        assumptions_recap = [
            f"{getattr(a, 'assumption_id', '')}: {getattr(a, 'assumption_statement', '')}"
            for a in framework.assumptions[:5]
        ]
        
        # Possible extensions
        extensions = [
            "Introduce heterogeneity across agents",
            "Add additional shocks or frictions",
            "Extend to open economy setting",
            "Incorporate learning or bounded rationality"
        ]
        
        formal_model = FormalMathematicalModel(
            model_title=model_title,
            model_summary=model_summary,
            based_on_framework=framework.framework_title,
            variables=variables,
            parameters=parameters,
            equations=equations,
            constraints=constraints,
            dynamics=dynamics,
            equilibrium=equilibrium,
            solution_method=solution_method,
            model_notation=notation,
            model_assumptions_recap=assumptions_recap,
            model_extensions=extensions,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_variables": len(variables),
                "num_parameters": len(parameters),
                "num_equations": len(equations),
                "num_constraints": len(constraints)
            }
        )
        
        return formal_model


# ========== ORCHESTRATOR ==========

class ModelDesignOrchestrator:
    """Orchestrator for the model design stage."""
    
    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.collector = collector
        self.model_designer = ModelDesigner(self.api_key, collector=collector)
        self.design_output: Optional[ModelDesignOutput] = None
    
    def run_design_pipeline(
        self,
        theory_output: Dict,
        revision_request: Optional[Dict] = None
    ) -> ModelDesignOutput:
        """Run the complete model design pipeline.

        ``revision_request`` (optional) is a data-feasibility Model Revision Request
        applied during a feasibility-loop cycle; None -> unchanged behavior.
        """
        
        print(f"\n{'='*70}")
        print(f"MODEL DESIGN PIPELINE")
        print(f"{'='*70}")
        
        # Parse theoretical frameworks
        frameworks_data = theory_output.get('theoretical_frameworks', [])
        frameworks = [TheoreticalFramework(**f) for f in frameworks_data]
        
        print(f"Theoretical Frameworks: {len(frameworks)}")
        print(f"{'='*70}\n")
        
        # Design formal models — independent per framework, so run them CONCURRENTLY (each design
        # is ~8 sequential LLM calls incl. a ~340s equation-formulation step; vLLM batches the
        # cross-model calls for ~7.5x aggregate throughput -> ~1 model's wall-time for all).
        _results = parallel_map(
            lambda fw: self.model_designer.design_formal_model(fw, revision_request=revision_request),
            frameworks,
        )
        formal_models = []
        for fw, r in zip(frameworks, _results):
            if isinstance(r, Exception):
                print(f"  [design] framework '{getattr(fw, 'framework_title', '?')}' failed: {r}")
                continue
            formal_models.append(r)
        
        # Create output
        self.design_output = ModelDesignOutput(
            theoretical_frameworks=[
                f.model_dump() if hasattr(f, "model_dump") else f for f in frameworks
            ],
            formal_models=formal_models,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_frameworks": len(frameworks),
                "num_models": len(formal_models)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Formal Models Designed: {len(formal_models)}")
        print(f"{'='*70}\n")
        
        return self.design_output
    
    def save_design_output(self, filename: str = "model_design_output.json"):
        """Save design output to JSON."""
        if not self.design_output:
            print("No design output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.design_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Design output saved to {filename}")
    
    def save_models_text(self, filename: str = "formal_mathematical_models.txt"):
        """Save formal models in readable text format."""
        if not self.design_output:
            print("No design output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FORMAL MATHEMATICAL MODELS\n")
            f.write("="*70 + "\n\n")
            
            for i, model in enumerate(self.design_output.formal_models, 1):
                f.write(f"MODEL {i}: {model.model_title}\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"SUMMARY:\n{model.model_summary}\n\n")
                
                f.write(f"NOTATION:\n{model.model_notation}\n\n")
                
                f.write(f"VARIABLES ({len(model.variables)}):\n")
                for var in model.variables:
                    f.write(f"  {var.variable_symbol}: {var.variable_name} ({var.variable_type})\n")
                    f.write(f"     Domain: {var.domain} | {var.description}\n\n")
                
                f.write(f"PARAMETERS ({len(model.parameters)}):\n")
                for param in model.parameters:
                    f.write(f"  {param.parameter_symbol}: {param.parameter_name}\n")
                    f.write(f"     Range: {param.typical_range} | {param.description}\n\n")
                
                f.write(f"EQUATIONS ({len(model.equations)}):\n")
                for eq in model.equations:
                    f.write(f"  {eq.equation_id}. {eq.equation_name} ({eq.equation_type})\n")
                    f.write(f"     LaTeX: {eq.equation_latex}\n")
                    f.write(f"     Plain: {eq.equation_plain}\n")
                    f.write(f"     Interpretation: {eq.interpretation}\n\n")
                
                f.write(f"CONSTRAINTS ({len(model.constraints)}):\n")
                for const in model.constraints:
                    f.write(f"  {const.constraint_id}. {const.constraint_type}\n")
                    f.write(f"     {const.constraint_latex}\n")
                    f.write(f"     {const.description}\n\n")
                
                f.write(f"DYNAMICS:\n")
                f.write(f"  Time Structure: {model.dynamics.time_structure}\n")
                f.write(f"  Time Horizon: {model.dynamics.time_horizon}\n")
                f.write(f"  State Variables: {', '.join(model.dynamics.state_variables)}\n")
                f.write(f"  Control Variables: {', '.join(model.dynamics.control_variables)}\n\n")
                
                f.write(f"EQUILIBRIUM:\n")
                f.write(f"  Type: {model.equilibrium.equilibrium_type}\n")
                f.write(f"  Conditions:\n")
                for cond in model.equilibrium.equilibrium_conditions:
                    f.write(f"    - {cond}\n")
                f.write("\n")
                
                f.write(f"SOLUTION METHOD:\n")
                f.write(f"  Approach: {model.solution_method.solution_approach}\n")
                f.write(f"  Steps:\n")
                for step in model.solution_method.solution_steps:
                    f.write(f"    {step}\n")
                f.write(f"  Software: {', '.join(model.solution_method.software_requirements)}\n\n")
                
                f.write("-"*70 + "\n\n")
        
        print(f"[Orchestrator] Models saved to {filename}")


def main():
    """Main function for model design stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load theory output
    theory_file = auto_input(
        "Enter path to theory output JSON (from Stage 1): ",
        default=get_default("theory_file_path"),
    ).strip()
    
    if not os.path.exists(theory_file):
        print(f"Error: {theory_file} not found!")
        return
    
    # Load data
    with open(theory_file, 'r', encoding='utf-8') as f:
        theory_output = json.load(f)
    
    # Run pipeline
    orchestrator = ModelDesignOrchestrator()
    design_output = orchestrator.run_design_pipeline(theory_output)
    
    # Save outputs
    orchestrator.save_design_output("model_design_output.json")
    orchestrator.save_models_text("formal_mathematical_models.txt")
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL DESIGN SUMMARY")
    print("="*70)
    for i, model in enumerate(design_output.formal_models, 1):
        print(f"\n{i}. {model.model_title}")
        print(f"   Variables: {len(model.variables)}")
        print(f"   Parameters: {len(model.parameters)}")
        print(f"   Equations: {len(model.equations)}")
        print(f"   Constraints: {len(model.constraints)}")
        print(f"   Solution: {model.solution_method.solution_approach}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
