"""
Populate Neo4j Aura database with residential construction data.
This script creates a hierarchical structure of Phases, Tasks, and Subtasks
for residential construction projects.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(".env")


# Residential construction data structure
RESIDENTIAL_CONSTRUCTION_DATA = {
    "Site Preparation": {
        "description": "Initial site work before construction begins",
        "tasks": [
            {
                "name": "Site Survey and Staking",
                "description": "Professional survey to mark property boundaries and building footprint",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Boundary survey",
                        "description": "Mark property lines and corners with stakes",
                    },
                    {
                        "name": "Elevation survey",
                        "description": "Determine ground levels and drainage patterns",
                    },
                    {
                        "name": "Building layout staking",
                        "description": "Mark foundation corners and reference points",
                    },
                ],
            },
            {
                "name": "Permits and Approvals",
                "description": "Obtain all necessary building permits and approvals",
                "duration_days": 14,
                "subtasks": [
                    {
                        "name": "Building permit application",
                        "description": "Submit building plans and apply for construction permit",
                    },
                    {
                        "name": "Zoning approval",
                        "description": "Verify compliance with local zoning regulations",
                    },
                    {
                        "name": "Utility connections approval",
                        "description": "Coordinate with utility companies for service connections",
                    },
                ],
            },
            {
                "name": "Site Clearing",
                "description": "Clear the construction site of obstacles",
                "duration_days": 3,
                "subtasks": [
                    {
                        "name": "Tree and vegetation removal",
                        "description": "Remove trees, shrubs, and vegetation from building area",
                    },
                    {
                        "name": "Debris removal",
                        "description": "Clear rocks, stumps, and existing structures",
                    },
                    {
                        "name": "Topsoil stripping",
                        "description": "Remove and stockpile topsoil for later landscaping",
                    },
                ],
            },
            {
                "name": "Excavation",
                "description": "Excavate for foundation and utilities",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Foundation excavation",
                        "description": "Dig foundation trenches or basement excavation",
                    },
                    {
                        "name": "Utility trenching",
                        "description": "Excavate trenches for water, sewer, and electrical lines",
                    },
                    {
                        "name": "Grading and leveling",
                        "description": "Level the building pad to proper elevation",
                    },
                ],
            },
        ],
    },
    "Foundation": {
        "description": "Construction of the building foundation",
        "tasks": [
            {
                "name": "Footings",
                "description": "Pour concrete footings to support foundation walls",
                "duration_days": 3,
                "subtasks": [
                    {
                        "name": "Footing formwork",
                        "description": "Install wooden forms for concrete footings",
                    },
                    {
                        "name": "Rebar installation",
                        "description": "Place reinforcing steel bars in footing forms",
                    },
                    {
                        "name": "Concrete pour",
                        "description": "Pour and finish concrete footings",
                    },
                ],
            },
            {
                "name": "Foundation Walls",
                "description": "Construct foundation walls on footings",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "Wall formwork",
                        "description": "Set up forms for poured concrete walls or lay CMU blocks",
                    },
                    {
                        "name": "Waterproofing",
                        "description": "Apply waterproof membrane to exterior foundation walls",
                    },
                    {
                        "name": "Drainage installation",
                        "description": "Install French drains and foundation drainage system",
                    },
                ],
            },
            {
                "name": "Slab Preparation",
                "description": "Prepare and pour concrete slab",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Gravel base",
                        "description": "Spread and compact gravel sub-base for slab",
                    },
                    {
                        "name": "Vapor barrier",
                        "description": "Install polyethylene vapor barrier over gravel",
                    },
                    {
                        "name": "Slab reinforcement",
                        "description": "Place wire mesh or rebar for slab reinforcement",
                    },
                    {
                        "name": "Slab pour",
                        "description": "Pour, level, and finish concrete slab",
                    },
                ],
            },
            {
                "name": "Backfill",
                "description": "Backfill around foundation walls",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Foundation inspection",
                        "description": "Complete foundation inspection before backfill",
                    },
                    {
                        "name": "Backfill placement",
                        "description": "Place and compact fill material around foundation",
                    },
                    {
                        "name": "Grade establishment",
                        "description": "Establish proper grade sloping away from foundation",
                    },
                ],
            },
        ],
    },
    "Framing": {
        "description": "Structural framing of walls, floors, and roof",
        "tasks": [
            {
                "name": "Floor Framing",
                "description": "Frame the floor system",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Sill plate installation",
                        "description": "Install pressure-treated sill plates on foundation",
                    },
                    {
                        "name": "Floor joist installation",
                        "description": "Install floor joists at proper spacing",
                    },
                    {
                        "name": "Subfloor installation",
                        "description": "Install plywood or OSB subfloor sheathing",
                    },
                ],
            },
            {
                "name": "Wall Framing",
                "description": "Frame exterior and interior walls",
                "duration_days": 7,
                "subtasks": [
                    {
                        "name": "Exterior wall framing",
                        "description": "Build and raise exterior wall frames with headers",
                    },
                    {
                        "name": "Interior wall framing",
                        "description": "Frame interior partition walls",
                    },
                    {
                        "name": "Wall sheathing",
                        "description": "Install exterior wall sheathing (OSB or plywood)",
                    },
                    {
                        "name": "Window and door openings",
                        "description": "Frame openings for windows and doors with proper headers",
                    },
                ],
            },
            {
                "name": "Ceiling and Roof Framing",
                "description": "Frame ceiling joists and roof structure",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "Ceiling joist installation",
                        "description": "Install ceiling joists or bottom chord of trusses",
                    },
                    {
                        "name": "Roof truss installation",
                        "description": "Set and brace pre-fabricated roof trusses",
                    },
                    {
                        "name": "Roof sheathing",
                        "description": "Install plywood or OSB roof deck",
                    },
                ],
            },
        ],
    },
    "Roofing": {
        "description": "Installation of roofing materials and weatherproofing",
        "tasks": [
            {
                "name": "Roof Underlayment",
                "description": "Install roofing underlayment and ice/water shield",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Drip edge installation",
                        "description": "Install metal drip edge at roof edges",
                    },
                    {
                        "name": "Ice and water shield",
                        "description": "Apply ice and water shield at eaves and valleys",
                    },
                    {
                        "name": "Felt underlayment",
                        "description": "Install roofing felt over entire roof deck",
                    },
                ],
            },
            {
                "name": "Shingle Installation",
                "description": "Install roofing shingles or other finish roofing",
                "duration_days": 3,
                "subtasks": [
                    {
                        "name": "Starter shingles",
                        "description": "Install starter strip shingles at eaves",
                    },
                    {
                        "name": "Field shingles",
                        "description": "Install main field of asphalt shingles",
                    },
                    {
                        "name": "Ridge cap installation",
                        "description": "Install ridge cap shingles at roof peaks",
                    },
                ],
            },
            {
                "name": "Roof Penetrations",
                "description": "Install flashing and vents",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Plumbing vent flashing",
                        "description": "Install flashing boots around plumbing vents",
                    },
                    {
                        "name": "Roof vents",
                        "description": "Install ridge vents or box vents for attic ventilation",
                    },
                    {
                        "name": "Chimney flashing",
                        "description": "Install step flashing and counter flashing at chimneys",
                    },
                ],
            },
        ],
    },
    "Plumbing Rough-In": {
        "description": "Installation of plumbing supply and drain lines",
        "tasks": [
            {
                "name": "Drain, Waste, and Vent (DWV)",
                "description": "Install drainage and vent piping",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Main drain line",
                        "description": "Install main building drain connected to sewer/septic",
                    },
                    {
                        "name": "Branch drain lines",
                        "description": "Install drain lines to each fixture location",
                    },
                    {
                        "name": "Vent stack installation",
                        "description": "Install vent pipes through roof for drainage venting",
                    },
                ],
            },
            {
                "name": "Water Supply Lines",
                "description": "Install hot and cold water supply piping",
                "duration_days": 3,
                "subtasks": [
                    {
                        "name": "Main water line",
                        "description": "Connect main water supply from street or well",
                    },
                    {
                        "name": "Hot and cold distribution",
                        "description": "Run copper or PEX supply lines to all fixtures",
                    },
                    {
                        "name": "Water heater connection",
                        "description": "Rough-in water heater supply and connections",
                    },
                ],
            },
            {
                "name": "Fixture Rough-In",
                "description": "Rough-in for all plumbing fixtures",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Bathroom rough-in",
                        "description": "Install supply and drain for toilets, sinks, showers",
                    },
                    {
                        "name": "Kitchen rough-in",
                        "description": "Rough-in for kitchen sink, dishwasher, ice maker",
                    },
                    {
                        "name": "Laundry rough-in",
                        "description": "Install washing machine supply and drain box",
                    },
                ],
            },
        ],
    },
    "Electrical Rough-In": {
        "description": "Installation of electrical wiring and boxes",
        "tasks": [
            {
                "name": "Main Panel Installation",
                "description": "Install main electrical service panel",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Panel mounting",
                        "description": "Mount main breaker panel in designated location",
                    },
                    {
                        "name": "Service entrance",
                        "description": "Install service entrance cables from meter to panel",
                    },
                    {
                        "name": "Grounding system",
                        "description": "Install grounding electrode and bonding connections",
                    },
                ],
            },
            {
                "name": "Branch Circuit Wiring",
                "description": "Run electrical circuits throughout the house",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "Outlet circuits",
                        "description": "Wire 120V circuits for outlets in each room",
                    },
                    {
                        "name": "Lighting circuits",
                        "description": "Run wiring for all light fixtures and switches",
                    },
                    {
                        "name": "Dedicated circuits",
                        "description": "Install 240V circuits for appliances (dryer, range, AC)",
                    },
                ],
            },
            {
                "name": "Low Voltage Wiring",
                "description": "Install communications and data wiring",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Network cabling",
                        "description": "Run CAT6 ethernet cables to key locations",
                    },
                    {
                        "name": "Cable/TV wiring",
                        "description": "Install coaxial cables for television service",
                    },
                    {
                        "name": "Doorbell and thermostat wiring",
                        "description": "Run low voltage wiring for doorbell and HVAC controls",
                    },
                ],
            },
        ],
    },
    "HVAC Installation": {
        "description": "Heating, ventilation, and air conditioning installation",
        "tasks": [
            {
                "name": "Ductwork Installation",
                "description": "Install HVAC ductwork throughout the house",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Main trunk lines",
                        "description": "Install main supply and return trunk ducts",
                    },
                    {
                        "name": "Branch ducts",
                        "description": "Run branch ducts to each room",
                    },
                    {
                        "name": "Register boots",
                        "description": "Install register boots for supply and return vents",
                    },
                ],
            },
            {
                "name": "Equipment Installation",
                "description": "Install HVAC equipment",
                "duration_days": 3,
                "subtasks": [
                    {
                        "name": "Furnace installation",
                        "description": "Install furnace or air handler unit",
                    },
                    {
                        "name": "AC condenser installation",
                        "description": "Set outdoor AC condenser unit on pad",
                    },
                    {
                        "name": "Refrigerant lines",
                        "description": "Run and connect refrigerant lines between units",
                    },
                ],
            },
            {
                "name": "Ventilation",
                "description": "Install exhaust and ventilation systems",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Bath exhaust fans",
                        "description": "Install bathroom exhaust fans and duct to exterior",
                    },
                    {
                        "name": "Range hood venting",
                        "description": "Install kitchen range hood exhaust duct",
                    },
                    {
                        "name": "Fresh air intake",
                        "description": "Install fresh air intake for combustion and ventilation",
                    },
                ],
            },
        ],
    },
    "Insulation": {
        "description": "Installation of thermal and sound insulation",
        "tasks": [
            {
                "name": "Wall Insulation",
                "description": "Insulate exterior walls",
                "duration_days": 3,
                "subtasks": [
                    {
                        "name": "Exterior wall batts",
                        "description": "Install fiberglass or mineral wool batts in exterior walls",
                    },
                    {
                        "name": "Air sealing",
                        "description": "Seal gaps around windows, doors, and penetrations",
                    },
                    {
                        "name": "Vapor barrier",
                        "description": "Install vapor retarder as required by climate zone",
                    },
                ],
            },
            {
                "name": "Attic Insulation",
                "description": "Insulate attic floor or roof",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Blown-in insulation",
                        "description": "Install blown-in insulation on attic floor",
                    },
                    {
                        "name": "Attic baffles",
                        "description": "Install ventilation baffles at eaves",
                    },
                    {
                        "name": "Attic hatch insulation",
                        "description": "Insulate and weatherstrip attic access hatch",
                    },
                ],
            },
            {
                "name": "Floor Insulation",
                "description": "Insulate floors over unconditioned spaces",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Crawlspace insulation",
                        "description": "Install insulation under floors over crawlspace",
                    },
                    {
                        "name": "Rim joist insulation",
                        "description": "Insulate and seal rim joists at foundation",
                    },
                    {
                        "name": "Garage ceiling insulation",
                        "description": "Insulate ceiling of attached garage",
                    },
                ],
            },
        ],
    },
    "Drywall": {
        "description": "Installation and finishing of drywall",
        "tasks": [
            {
                "name": "Drywall Hanging",
                "description": "Install drywall panels on walls and ceilings",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Ceiling drywall",
                        "description": "Hang drywall on ceilings first",
                    },
                    {
                        "name": "Wall drywall",
                        "description": "Install drywall on walls with proper fastening",
                    },
                    {
                        "name": "Specialty areas",
                        "description": "Install moisture-resistant drywall in wet areas",
                    },
                ],
            },
            {
                "name": "Drywall Finishing",
                "description": "Tape, mud, and sand drywall joints",
                "duration_days": 6,
                "subtasks": [
                    {
                        "name": "Taping",
                        "description": "Apply tape and first coat of joint compound",
                    },
                    {
                        "name": "Second coat",
                        "description": "Apply second coat of joint compound",
                    },
                    {
                        "name": "Final coat and sanding",
                        "description": "Apply finish coat and sand smooth",
                    },
                ],
            },
            {
                "name": "Texture and Prime",
                "description": "Apply texture and primer to drywall",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Ceiling texture",
                        "description": "Apply texture to ceilings as specified",
                    },
                    {
                        "name": "Wall texture",
                        "description": "Apply texture to walls if specified",
                    },
                    {
                        "name": "Primer coat",
                        "description": "Apply drywall primer to all surfaces",
                    },
                ],
            },
        ],
    },
    "Interior Finishes": {
        "description": "Interior trim, flooring, and finish work",
        "tasks": [
            {
                "name": "Interior Doors and Trim",
                "description": "Install interior doors and millwork",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "Door hanging",
                        "description": "Install pre-hung interior doors",
                    },
                    {
                        "name": "Door casing",
                        "description": "Install door trim and casing",
                    },
                    {
                        "name": "Baseboard installation",
                        "description": "Install baseboard molding throughout",
                    },
                    {
                        "name": "Window trim",
                        "description": "Install window casing and stools",
                    },
                ],
            },
            {
                "name": "Flooring Installation",
                "description": "Install finish flooring materials",
                "duration_days": 6,
                "subtasks": [
                    {
                        "name": "Hardwood flooring",
                        "description": "Install and finish hardwood flooring",
                    },
                    {
                        "name": "Tile flooring",
                        "description": "Install ceramic or porcelain tile in wet areas",
                    },
                    {
                        "name": "Carpet installation",
                        "description": "Install carpet in bedrooms and living areas",
                    },
                ],
            },
            {
                "name": "Painting",
                "description": "Interior painting and finishing",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "Wall painting",
                        "description": "Apply finish paint to walls",
                    },
                    {
                        "name": "Trim painting",
                        "description": "Paint doors, trim, and millwork",
                    },
                    {
                        "name": "Touch-up",
                        "description": "Final paint touch-up after other trades complete",
                    },
                ],
            },
            {
                "name": "Cabinets and Countertops",
                "description": "Install kitchen and bath cabinetry",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Cabinet installation",
                        "description": "Install kitchen and bathroom cabinets",
                    },
                    {
                        "name": "Countertop installation",
                        "description": "Install granite, quartz, or laminate countertops",
                    },
                    {
                        "name": "Hardware installation",
                        "description": "Install cabinet knobs, pulls, and hinges",
                    },
                ],
            },
        ],
    },
    "Exterior Finishes": {
        "description": "Exterior siding, trim, and finish work",
        "tasks": [
            {
                "name": "Siding Installation",
                "description": "Install exterior wall cladding",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "House wrap",
                        "description": "Install weather-resistant barrier (Tyvek or similar)",
                    },
                    {
                        "name": "Siding installation",
                        "description": "Install vinyl, fiber cement, or wood siding",
                    },
                    {
                        "name": "Corner and trim pieces",
                        "description": "Install corner boards and trim accessories",
                    },
                ],
            },
            {
                "name": "Exterior Trim and Paint",
                "description": "Finish exterior millwork and painting",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Fascia and soffit",
                        "description": "Install fascia boards and soffit panels",
                    },
                    {
                        "name": "Exterior caulking",
                        "description": "Caulk all exterior joints and penetrations",
                    },
                    {
                        "name": "Exterior painting",
                        "description": "Paint or stain exterior trim and siding",
                    },
                ],
            },
            {
                "name": "Concrete Flatwork",
                "description": "Install driveways, walks, and patios",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Driveway pour",
                        "description": "Form and pour concrete driveway",
                    },
                    {
                        "name": "Sidewalk installation",
                        "description": "Pour concrete sidewalks and walkways",
                    },
                    {
                        "name": "Patio construction",
                        "description": "Build patio with concrete, pavers, or deck",
                    },
                ],
            },
            {
                "name": "Landscaping",
                "description": "Final grading and landscaping",
                "duration_days": 5,
                "subtasks": [
                    {
                        "name": "Final grading",
                        "description": "Final grade lot for proper drainage",
                    },
                    {
                        "name": "Topsoil and seed/sod",
                        "description": "Spread topsoil and install lawn",
                    },
                    {
                        "name": "Planting",
                        "description": "Install trees, shrubs, and plants",
                    },
                ],
            },
        ],
    },
    "Final Inspections": {
        "description": "Final inspections, testing, and punch list",
        "tasks": [
            {
                "name": "MEP Final",
                "description": "Final mechanical, electrical, plumbing inspections",
                "duration_days": 2,
                "subtasks": [
                    {
                        "name": "Electrical final",
                        "description": "Complete electrical final inspection",
                    },
                    {
                        "name": "Plumbing final",
                        "description": "Complete plumbing final inspection and test",
                    },
                    {
                        "name": "HVAC final",
                        "description": "Complete HVAC final inspection and commissioning",
                    },
                ],
            },
            {
                "name": "Building Final",
                "description": "Final building inspection for certificate of occupancy",
                "duration_days": 1,
                "subtasks": [
                    {
                        "name": "Final walkthrough",
                        "description": "Complete final inspection walkthrough with inspector",
                    },
                    {
                        "name": "Certificate of occupancy",
                        "description": "Obtain certificate of occupancy from building department",
                    },
                ],
            },
            {
                "name": "Punch List",
                "description": "Complete punch list items and final cleaning",
                "duration_days": 4,
                "subtasks": [
                    {
                        "name": "Punch list walkthrough",
                        "description": "Walk through with owner to identify punch list items",
                    },
                    {
                        "name": "Punch list completion",
                        "description": "Complete all punch list items",
                    },
                    {
                        "name": "Final cleaning",
                        "description": "Professional cleaning of entire house",
                    },
                ],
            },
            {
                "name": "Owner Orientation",
                "description": "Homeowner orientation and handover",
                "duration_days": 1,
                "subtasks": [
                    {
                        "name": "Systems training",
                        "description": "Train homeowner on all home systems and equipment",
                    },
                    {
                        "name": "Documentation handover",
                        "description": "Provide manuals, warranties, and as-built documents",
                    },
                    {
                        "name": "Key handover",
                        "description": "Final key handover and project closeout",
                    },
                ],
            },
        ],
    },
}


def create_database_schema(driver):
    """Create constraints and indexes for the schema."""
    with driver.session() as session:
        # Create constraints for uniqueness
        constraints = [
            "CREATE CONSTRAINT phase_name IF NOT EXISTS FOR (p:Phase) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT subtask_id IF NOT EXISTS FOR (s:Subtask) REQUIRE s.id IS UNIQUE",
        ]
        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"✓ Created constraint")
            except Exception as e:
                print(f"⚠ Constraint may already exist: {e}")


def clear_existing_data(driver):
    """Clear all existing nodes and relationships."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing data")


def populate_database(driver):
    """Populate the database with residential construction data."""
    with driver.session() as session:
        phase_order = 0
        for phase_name, phase_data in RESIDENTIAL_CONSTRUCTION_DATA.items():
            phase_order += 1

            # Create Phase node
            session.run(
                """
                CREATE (p:Phase {
                    name: $name,
                    description: $description,
                    order: $order
                })
                """,
                name=phase_name,
                description=phase_data["description"],
                order=phase_order,
            )
            print(f"✓ Created Phase: {phase_name}")

            task_order = 0
            prev_task_id = None

            for task in phase_data["tasks"]:
                task_order += 1
                task_id = f"{phase_name}_{task['name']}".replace(" ", "_").lower()

                # Create Task node and connect to Phase
                session.run(
                    """
                    MATCH (p:Phase {name: $phase_name})
                    CREATE (t:Task {
                        id: $task_id,
                        name: $task_name,
                        description: $description,
                        duration_days: $duration,
                        phase: $phase_name,
                        order: $order
                    })
                    CREATE (p)-[:HAS_TASK]->(t)
                    """,
                    phase_name=phase_name,
                    task_id=task_id,
                    task_name=task["name"],
                    description=task["description"],
                    duration=task["duration_days"],
                    order=task_order,
                )
                print(f"  ✓ Created Task: {task['name']}")

                # Create dependency to previous task in same phase
                if prev_task_id:
                    session.run(
                        """
                        MATCH (t1:Task {id: $prev_id})
                        MATCH (t2:Task {id: $curr_id})
                        CREATE (t2)-[:DEPENDS_ON {relationship: 'FS', lag_days: 0}]->(t1)
                        """,
                        prev_id=prev_task_id,
                        curr_id=task_id,
                    )

                prev_task_id = task_id

                # Create Subtask nodes
                subtask_order = 0
                for subtask in task["subtasks"]:
                    subtask_order += 1
                    subtask_id = f"{task_id}_{subtask['name']}".replace(
                        " ", "_"
                    ).lower()

                    session.run(
                        """
                        MATCH (t:Task {id: $task_id})
                        CREATE (s:Subtask {
                            id: $subtask_id,
                            name: $subtask_name,
                            description: $description,
                            task: $task_name,
                            phase: $phase_name,
                            order: $order
                        })
                        CREATE (t)-[:HAS_SUBTASK]->(s)
                        """,
                        task_id=task_id,
                        subtask_id=subtask_id,
                        subtask_name=subtask["name"],
                        description=subtask["description"],
                        task_name=task["name"],
                        phase_name=phase_name,
                        order=subtask_order,
                    )
                print(f"    ✓ Created {len(task['subtasks'])} subtasks")

        # Create cross-phase dependencies
        cross_phase_deps = [
            (
                "site_preparation_backfill",
                "foundation_footings",
            ),  # Site Prep -> Foundation
            ("foundation_backfill", "framing_floor_framing"),  # Foundation -> Framing
            (
                "framing_ceiling_and_roof_framing",
                "roofing_roof_underlayment",
            ),  # Framing -> Roofing
            (
                "framing_wall_framing",
                "plumbing_rough-in_drain,_waste,_and_vent_(dwv)",
            ),  # Framing -> Plumbing
            (
                "framing_wall_framing",
                "electrical_rough-in_main_panel_installation",
            ),  # Framing -> Electrical
            (
                "framing_wall_framing",
                "hvac_installation_ductwork_installation",
            ),  # Framing -> HVAC
            (
                "plumbing_rough-in_fixture_rough-in",
                "insulation_wall_insulation",
            ),  # Plumbing -> Insulation
            (
                "electrical_rough-in_low_voltage_wiring",
                "insulation_wall_insulation",
            ),  # Electrical -> Insulation
            (
                "hvac_installation_ventilation",
                "insulation_wall_insulation",
            ),  # HVAC -> Insulation
            (
                "insulation_floor_insulation",
                "drywall_drywall_hanging",
            ),  # Insulation -> Drywall
            (
                "drywall_texture_and_prime",
                "interior_finishes_interior_doors_and_trim",
            ),  # Drywall -> Interior
            (
                "roofing_roof_penetrations",
                "exterior_finishes_siding_installation",
            ),  # Roofing -> Exterior
            (
                "interior_finishes_painting",
                "final_inspections_mep_final",
            ),  # Interior -> Final
            (
                "exterior_finishes_landscaping",
                "final_inspections_building_final",
            ),  # Exterior -> Final
        ]

        print("\n✓ Creating cross-phase dependencies...")
        for prev_id, curr_id in cross_phase_deps:
            try:
                # Fix the task IDs to match actual format
                prev_id_fixed = (
                    prev_id.replace(",", "").replace("(", "").replace(")", "")
                )
                curr_id_fixed = (
                    curr_id.replace(",", "").replace("(", "").replace(")", "")
                )

                session.run(
                    """
                    MATCH (t1:Task) WHERE t1.id CONTAINS $prev_partial
                    MATCH (t2:Task) WHERE t2.id CONTAINS $curr_partial
                    MERGE (t2)-[:DEPENDS_ON {relationship: 'FS', lag_days: 0}]->(t1)
                    """,
                    prev_partial=prev_id.split("_")[1] if "_" in prev_id else prev_id,
                    curr_partial=curr_id.split("_")[1] if "_" in curr_id else curr_id,
                )
            except Exception as e:
                print(f"  ⚠ Could not create dependency {prev_id} -> {curr_id}: {e}")


def get_statistics(driver):
    """Get and print database statistics."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Phase) WITH count(p) as phases
            MATCH (t:Task) WITH phases, count(t) as tasks
            MATCH (s:Subtask) WITH phases, tasks, count(s) as subtasks
            MATCH ()-[r:HAS_TASK]->() WITH phases, tasks, subtasks, count(r) as task_rels
            MATCH ()-[r:HAS_SUBTASK]->() WITH phases, tasks, subtasks, task_rels, count(r) as subtask_rels
            MATCH ()-[r:DEPENDS_ON]->() WITH phases, tasks, subtasks, task_rels, subtask_rels, count(r) as deps
            RETURN phases, tasks, subtasks, task_rels, subtask_rels, deps
        """
        )
        stats = result.single()

        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        print(f"Phases:              {stats['phases']}")
        print(f"Tasks:               {stats['tasks']}")
        print(f"Subtasks:            {stats['subtasks']}")
        print(f"Phase->Task rels:    {stats['task_rels']}")
        print(f"Task->Subtask rels:  {stats['subtask_rels']}")
        print(f"Dependencies:        {stats['deps']}")
        print("=" * 50)


def main():
    # Get connection details from environment
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        print("❌ Missing Neo4j connection details in .env file")
        return

    print(f"Connecting to Neo4j at {uri}...")

    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        # Verify connection
        driver.verify_connectivity()
        print("✓ Connected to Neo4j successfully!\n")

        # Ask for confirmation before clearing
        print("⚠️  This will clear ALL existing data in the database.")
        confirm = input("Continue? (yes/no): ").strip().lower()

        if confirm != "yes":
            print("Cancelled.")
            return

        print("\n" + "=" * 50)
        print("POPULATING DATABASE")
        print("=" * 50 + "\n")

        # Create schema
        create_database_schema(driver)

        # Clear existing data
        clear_existing_data(driver)

        # Populate with new data
        populate_database(driver)

        # Show statistics
        get_statistics(driver)

        print("\n✅ Database population complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    main()
