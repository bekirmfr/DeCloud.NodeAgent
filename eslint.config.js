import js from '@eslint/js';

export default [
    // ESLint recommended rules
    js.configs.recommended,
    
    {
        // Files to lint
        files: ['src/**/*.js', '*.js'],
        
        // Files to ignore
        ignores: [
            'node_modules/**',
            'src/DeCloud.NodeAgent/CloudInit/**',
            'src/DeCloud.NodeAgent/bin/**',
            '*.min.js'
        ],
        
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: 'module',
            globals: {
                // Browser globals
                window: 'readonly',
                document: 'readonly',
                console: 'readonly',
                fetch: 'readonly',
                localStorage: 'readonly',
                sessionStorage: 'readonly',
                FormData: 'readonly',
                URL: 'readonly',
                URLSearchParams: 'readonly',
                
                // Modern Web APIs
                WebSocket: 'readonly',
                crypto: 'readonly',
                navigator: 'readonly',
                
                // Vite globals
                import: 'readonly'
            }
        },
        
        rules: {
            // ================================================
            // SECURITY-FIRST RULES
            // ================================================
            'no-eval': 'error',                    // Prevent eval() - major security risk
            'no-implied-eval': 'error',            // Prevent setTimeout/setInterval with strings
            'no-new-func': 'error',                // Prevent Function constructor
            'no-script-url': 'error',              // Prevent javascript: URLs
            'no-alert': 'warn',                    // Discourage alert() in production
            
            // ================================================
            // CODE QUALITY RULES
            // ================================================
            'no-unused-vars': ['warn', {
                argsIgnorePattern: '^_',            // Allow unused args starting with _
                varsIgnorePattern: '^_'
            }],
            'no-undef': 'error',                   // Catch undefined variables
            'no-console': 'off',                   // Allow console for debugging
            'prefer-const': 'warn',                // Use const when possible
            'no-var': 'warn',                      // Prefer let/const over var
            'eqeqeq': ['warn', 'always'],          // Use === instead of ==
            'curly': ['warn', 'all'],              // Require curly braces
            'no-debugger': 'warn',                 // Warn about debugger statements
            
            // ================================================
            // ASYNC/PROMISE RULES
            // ================================================
            'no-async-promise-executor': 'error',  // Prevent common async/promise mistakes
            'require-await': 'off',                // Allow async functions without await
            
            // ================================================
            // BEST PRACTICES
            // ================================================
            'no-duplicate-imports': 'error',       // Prevent duplicate imports
            'no-useless-return': 'warn',           // Remove unnecessary returns
            'no-unreachable': 'error',             // Catch unreachable code
            'no-fallthrough': 'warn',              // Catch missing break in switch
        }
    }
];
