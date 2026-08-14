import js from "@eslint/js";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";
import solid from "eslint-plugin-solid";
import globals from "globals";

export default defineConfig([
    {
        ignores: ["dist", ".output", ".vinxi", ".wrangler", "node_modules", ".git"],
    },
    js.configs.recommended,
    ...tseslint.configs.recommended,
    {
        files: ["**/*.{ts,tsx}"],
        ...solid.configs["flat/typescript"],
        languageOptions: {
            globals: globals.browser,
        },
    },
    {
        files: ["**/*.tsx"],
        // Solid assigns element refs through compiled JSX, which ESLint's core
        // no-unassigned-vars rule cannot observe.
        rules: {
            "no-unassigned-vars": "off",
        },
    },
]);
