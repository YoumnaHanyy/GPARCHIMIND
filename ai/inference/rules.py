
RULES = {
    "Factory": {
        "core": [
            "UNKNOWN_OBJECT_TYPE",
            "RUNTIME_OBJECT_SELECTION"
        ],
        "generic": [
            "LOW_COUPLING",
            "EXTENSIBILITY",
            "OBJECT_REUSE",
            "HIGH_SCALABILITY",
            "HIGH_EXTENSIBILITY"  
        ]
    },

    "Builder": {
        "core": [
            "COMPLEX_OBJECT_CREATION",
            "STEP_BY_STEP_CREATION",
            "SEPARATION_CONSTRUCTION_REPRESENTATION",
            "MULTIPLE_REPRESENTATIONS",
            "CONTROLLED_CONSTRUCTION_PROCESS"
        ],
        "generic": [
            "SEPARATION_OF_DECISIONS",
            "CONTEXT_CUSTOMIZATION",
            "HIGH_REUSABILITY",
            "CONFIGURABLE_CONSTRUCTION"
        ]
    },

    "Singleton": {
        "core": [
            "SINGLE_INSTANCE",
            "GLOBAL_ACCESS_POINT",
            "CONTROLLED_INSTANTIATION"
        ],
        "generic": [
            "SHARED_INSTANCE",
            "RESOURCE_CONTROL",
            "SHARED_RESOURCE_MANAGEMENT",
            "LAZY_INITIALIZATION",
            "THREAD_SAFE_ACCESS"
        ]
    },

    "Observer": {
        "core": [
            "EVENT_DRIVEN",
            "ONE_TO_MANY_DEPENDENCY",
            "AUTO_UPDATE",
            "SUBSCRIBE_MECHANISM"
        ],
        "generic": [
            "REAL_TIME_UPDATES",
            "LOW_COUPLING_REQUIRED",
            "MULTIPLE_VIEWS",
            "UNKNOWN_RECEIVERS",
            "REAL_TIME"
        ]
    },

    "Strategy": {
        "core": [
            "MULTIPLE_ALGORITHMS",
            "RUNTIME_SWITCHING",
            "INTERCHANGEABLE_BEHAVIOR",
            "CONTEXT_DEPENDENT",
            "CONTEXT_USES_STRATEGY",
            "CLIENT_SELECTS_STRATEGY",
             "DYNAMIC_BEHAVIOR"  

        ],
        "generic": [
            "RULE_BASED_SELECTION",
            "EXTENSIBILITY",
            "AVOID_MODIFICATION",
            "ENCAPSULATE_VOLATILE_CODE",
            "USER_CHOICE",
            "COMMON_INTERFACE",
            "REUSABILITY",
            "AVOID_SPAGHETTI_CODE",
            "DELEGATION_TO_STRATEGY",
            "COMPOSITION_OVER_INHERITANCE",
            "CODE_GROWTH_PROBLEM"
        ]
    },

    "Mediator": {
        "core": [
            "NO_DIRECT_COMMUNICATION",
            "CENTRALIZED_CONTROL",
            "ENCAPSULATE_INTERACTIONS",
            "MEDIATOR_DEPENDENCY_ONLY",
            "MESSAGE_ROUTING",
            "MANAGE_COLLABORATION"
        ],
        "generic": [
            "LOOSE_COUPLING",
            "IMPROVE_MAINTAINABILITY",
            "HANDLE_COMPLEX_INTERACTIONS",
            "CENTRALIZED_DEPENDENCY",
            "TRADE_OFF_COUPLING",
            "REDUCE_COMMUNICATION_OVERHEAD",
            "MODULARITY" 
        ]
    },

    "Command": {
        "core": [
            "ENCAPSULATE_REQUEST",
            "DECOUPLE_INVOKER_RECEIVER",
            "PARAMETERIZE_REQUESTS",
            "COMMAND_COMPOSITION",
            "REPRESENT_OPERATIONS_AS_OBJECTS",
            "SELF_CONTAINED_REQUEST"
        ],
        "generic": [
            "COMMON_INTERFACE",
            "EXTENSIBILITY",
            "UNDO_REDO_SUPPORT",
            "COMMAND_STACK",
            "STORE_STATE_FOR_UNDO",
            "SEPARATION_OF_CONCERNS",
            "INVOKER_ABSTRACTION",
            "DEFERRED_EXECUTION",
            "SELF_CONFIGURATION",
            "AVOID_CONCRETE_DEPENDENCY",
            "AVOID_INVOKER_CONFIGURATION",
            "AVOID_LONG_PARAMETER_LIST",
            "AVOID_GENERIC_PARAMETER_HACK",
            "AVOID_IMPLICIT_COUPLING",
            "COMMAND_PROCESSOR",
            "SEPARATE_INVOCATION_LAYER",
            "UNDO_MANAGER",
            "PERSISTENT_HISTORY",
            "GROUP_COMMANDS",
            "SEPARATE_INPUT_FROM_EXECUTION",
            "EVENT_DRIVEN_COMMANDS",
            "DYNAMIC_PLUGIN_COMMANDS",
            "SUB_COMMANDS",
            "COMMAND_QUEUE",
            "COMMAND_LOGGING",
            "INVOKER_STORES_COMMANDS",
            "UNIFIED_OPERATION_HANDLING",
            "SERVICE_OPERATION_ENCAPSULATION"
        ]
    },

    "Adapter": {
    "core": [
        "INCOMPATIBLE_INTERFACES",
        "LEGACY_INTEGRATION",
        "INTERFACE_CONVERSION"
    ],
    "generic": [
        "REUSE_EXISTING_CODE",
        "SYSTEM_INTEGRATION",
        "WRAPPER_USAGE",
        "COMPATIBILITY_LAYER"
    ]
},
"Decorator": {
    "core": [
        "DYNAMIC_BEHAVIOR_EXTENSION",
        "RUNTIME_FEATURE_ADDITION",
        "WRAP_OBJECT"
    ],
    "generic": [
        "OPEN_CLOSED_PRINCIPLE",
        "FLEXIBLE_FEATURE_EXTENSION",
        "AVOID_SUBCLASSING",
        "COMPOSITION_OVER_INHERITANCE"
    ]
},
"Facade": {
    "core": [
        "SIMPLIFIED_INTERFACE",
        "COMPLEX_SYSTEM",
        "SUBSYSTEM_ABSTRACTION"
    ],
    "generic": [
        "REDUCE_COMPLEXITY",
        "EASY_USAGE",
        "LAYERED_ACCESS",
        "DECOUPLE_CLIENT"
    ]
},
"Proxy": {
    "core": [
        "CONTROLLED_ACCESS",
        "REMOTE_ACCESS",
        "LAZY_LOADING"
    ],
    "generic": [
        "SECURITY_CONTROL",
        "PERFORMANCE_OPTIMIZATION",
        "ACCESS_WRAPPER",
        "RESOURCE_MANAGEMENT"
    ]
},
"State": {
    "core": [
        "STATE_DEPENDENT_BEHAVIOR",
        "STATE_TRANSITIONS",
        "OBJECT_CHANGES_BEHAVIOR"
    ],
    "generic": [
        "AVOID_IF_ELSE_COMPLEXITY",
        "ENCAPSULATE_STATE_LOGIC",
        "CLEAN_STATE_MANAGEMENT",
        "DYNAMIC_BEHAVIOR"
    ]
},

"TemplateMethod": {
    "core": [
        "ALGORITHM_STRUCTURE",
        "STEP_BY_STEP_PROCESS",
        "FIXED_FLOW"
    ],
    "generic": [
        "CODE_REUSE",
        "EXTEND_STEPS",
        "BASE_CLASS_CONTROL",
        "STANDARDIZED_PROCESS"
    ]
},

}
