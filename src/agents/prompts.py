def single_COT_prompt(target_words, enemy_words, neutral_words, assassin, board_state, shot = 0):
    if shot == 0:
        prompt = f"""
                You are a STRICT Codenames Spymaster.

                Your job is to generate a VALID clue.

                ====================
                TASK
                ====================
                Select a subset of TARGET WORDS (min 1, max 8) that share a strong semantic connection.
                Then produce ONE clue word that links them.

                ====================
                INPUT
                ====================
                TARGET WORDS:
                {', '.join(target_words)}

                DANGEROUS WORDS (MUST AVOID):
                {', '.join(enemy_words + neutral_words + assassin)}

                ALL BOARD WORDS:
                {', '.join([item['word'] for item in board_state if not item['revealed']])}

                ====================
                STRICT RULES (MUST FOLLOW)
                ====================
                1. Output EXACTLY: <one_word> <number>
                - NO quotes, NO punctuation, NO extra text

                2. Clue must be ONE WORD only
                - No spaces
                - No hyphens
                - No symbols
                - Letters only (a-z)

                3. Clue MUST NOT:
                - Be equal to ANY board word
                - Be a substring of ANY board word
                - Contain ANY board word as substring

                4. Clue MUST relate to ALL chosen target words

                5. Clue MUST NOT relate to ANY dangerous word
                - If it could relate → REJECT it

                6. Prefer SPECIFIC clues over generic ones

                ====================
                PROCESS (FOLLOW STEP BY STEP)
                ====================
                Step 1: Propose a candidate group of target words
                Step 2: Think of a candidate clue
                Step 3: VALIDATE the clue:
                    - Check all RULES above
                    - If ANY rule is violated → discard and retry
                Step 4: Output ONLY the final valid clue

                ====================
                FINAL OUTPUT (ONLY THIS FORMAT)
                ====================
                clue number
                """
    elif shot == 1:
        prompt = f"""
                You are a STRICT Codenames Spymaster.

                Your job is to generate a VALID clue.

                ====================
                TASK
                ====================
                Select a subset of TARGET WORDS (min 1, max 8) that share a strong semantic connection.
                Then produce ONE clue word that links them.

                ====================
                INPUT
                ====================
                TARGET WORDS:
                {', '.join(target_words)}

                DANGEROUS WORDS (MUST AVOID):
                {', '.join(enemy_words + neutral_words + assassin)}

                ALL BOARD WORDS:
                {', '.join([item['word'] for item in board_state if not item['revealed']])}

                ====================
                STRICT RULES (MUST FOLLOW)
                ====================
                1. Output EXACTLY: <one_word> <number>
                - Example: pet 3
                - NO quotes, NO punctuation, NO extra text

                2. Clue must be ONE WORD only
                - No spaces
                - No hyphens
                - No symbols
                - Letters only (a-z)

                3. Clue MUST NOT:
                - Be equal to ANY board word
                - Be a substring of ANY board word
                - Contain ANY board word as substring

                4. Clue MUST relate to ALL chosen target words

                5. Clue MUST NOT relate to ANY dangerous word
                - If it could relate → REJECT it

                6. Prefer SPECIFIC clues over generic ones
                - GOOD: "pet"
                - BAD: "thing", "object", "entity"

                ====================
                PROCESS (FOLLOW STEP BY STEP)
                ====================
                Step 1: Propose a candidate group of target words
                Step 2: Think of a candidate clue
                Step 3: VALIDATE the clue:
                    - Check all RULES above
                    - If ANY rule is violated → discard and retry
                Step 4: Output ONLY the final valid clue

                ====================
                EXAMPLE
                ====================
                Target: cat, dog, mouse, car, tree
                Dangerous: truck, forest

                Valid:
                pet 3

                Invalid:
                animal 3  (too generic, may link to dangerous)
                cat 2     (board word)
                pet-animal 3 (invalid format)

                ====================
                FINAL OUTPUT (ONLY THIS FORMAT)
                ====================
                clue number
                """
    return prompt



class double_COT_prompt():
    def propose_groups(target_words, enemy_words, neutral_words, assassin_words, shot = 0):
        if shot == 0:
            prompt =f"""
                You are a STRICT Codenames Spymaster.

                Your job is to identify semantic structure in the target words.

                ====================
                TASK
                ====================
                Select the BEST subset of TARGET WORDS (1 to 8 words) that share a strong semantic connection. Maximize the number of target words covered by ONE clue,
                while MINIMIZING risk of association with dangerous words.

                You must balance:
                - Coverage (more words is better)
                - Safety (no overlap with dangerous words)
                - Prefer the largest coherent group possible
                - BUT never force weak or ambiguous connections
                - Only select words that clearly belong together

                You are penalized for:
                - Groups of size 1 unless NO safe multi-word group exists
                - Too small groups
                - Groups that could relate to dangerous words

                ====================
                INPUT
                ====================
                TARGET WORDS:
                {', '.join(target_words)}

                DANGEROUS WORDS (MUST AVOID ASSOCIATION WITH):
                {', '.join(enemy_words + neutral_words + assassin_words)}

                ====================
                STRICT RULES (MUST FOLLOW)
                ====================
                1. You MUST ONLY select words from TARGET WORDS

                2. Group must be semantically coherent:
                - All words must belong to the SAME concept category or theme
                - Do NOT mix unrelated concepts

                3. Avoid dangerous semantic leakage:
                - Do NOT form a group that could relate to dangerous words
                - If uncertain → exclude the word

                4. Group size:
                - Minimum 1 word
                - Maximum 8 words

                ====================
                PROCESS (FOLLOW STEP BY STEP)
                ====================
                Step 1: Scan target words for strong semantic clusters
                Step 2: Evaluate each possible cluster for coherence strength
                Step 3: Reject any cluster that overlaps with dangerous words
                Step 4: Select the BEST remaining cluster

                ====================
                OUTPUT FORMAT (STRICT JSON ONLY)
                ====================
                {{
                "group": ["word1", "word2", "word3"]
                }}
                """
        elif shot == 1:
            prompt = f"""
                You are a STRICT Codenames Spymaster.

                Your job is to identify semantic structure in the target words.

                ====================
                TASK
                ====================
                Select the BEST subset of TARGET WORDS (1 to 8 words) that share a strong semantic connection.

                OBJECTIVE:
                Maximize the number of target words covered by ONE clue,
                while MINIMIZING risk of association with dangerous words.

                You must balance:
                - Coverage (more words is better)
                - Safety (no overlap with dangerous words)
                - Prefer the largest coherent group possible
                - BUT never force weak or ambiguous connections
                - Only select words that clearly belong together

                You are penalized for:
                - Groups of size 1 unless NO safe multi-word group exists
                - Too small groups
                - Groups that could relate to dangerous words

                ====================
                EXAMPLES
                ====================

                Example 1:
                TARGET WORDS:
                cat, dog, mouse, car, tree

                DANGEROUS WORDS:
                truck, forest

                Analysis:
                - cat, dog, mouse → animals (strong cluster)
                - car → unrelated
                - tree → unrelated and close to forest (danger)

                Best group:
                ["cat", "dog", "mouse"]

                Output:
                {{"group": ["cat", "dog", "mouse"]}}

                --------------------

                Example 2:
                TARGET WORDS:
                apple, banana, orange, table, chair

                DANGEROUS WORDS:
                fruit, wood

                Analysis:
                - apple, banana, orange → fruits (BUT dangerous word "fruit" exists → risky)
                - table, chair → furniture (safe and coherent)

                Best group:
                ["table", "chair"]

                Output:
                {{"group": ["table", "chair"]}}

                --------------------

                Example 3:
                TARGET WORDS:
                helicopter, airplane, train, car, kite

                DANGEROUS WORDS:
                vehicle, road

                Analysis:
                - helicopter, airplane → flying vehicles (safe subset)
                - train, car → land vehicles (close to "road" → risky)
                - kite → weak relation

                Best group:
                ["helicopter", "airplane"]

                Output:
                {{"group": ["helicopter", "airplane"]}}

                --------------------

                Example 4 (BAD EXAMPLE - DO NOT DO THIS):
                TARGET WORDS:
                lion, tiger, table, pencil

                DANGEROUS WORDS:
                animal

                Bad grouping:
                ["lion", "tiger"]

                Reason:
                - Although coherent, it strongly relates to dangerous word "animal"

                Correct safer choice:
                ["table", "pencil"]

                Output:
                {{"group": ["table", "pencil"]}}

                ====================
                INPUT
                ====================
                TARGET WORDS:
                {', '.join(target_words)}

                DANGEROUS WORDS (MUST AVOID ASSOCIATION WITH):
                {', '.join(enemy_words + neutral_words + assassin_words)}

                ====================
                STRICT RULES (MUST FOLLOW)
                ====================
                1. You MUST ONLY select words from TARGET WORDS

                2. Group must be semantically coherent:
                - All words must belong to the SAME concept category or theme
                - Do NOT mix unrelated concepts

                3. Avoid dangerous semantic leakage:
                - Do NOT form a group that could relate to dangerous words
                - If uncertain → exclude the word

                4. Group size:
                - Minimum 1 word
                - Maximum 8 words

                ====================
                PROCESS (FOLLOW STEP BY STEP)
                ====================
                Step 1: Scan target words for strong semantic clusters
                Step 2: Evaluate each cluster for BOTH size and safety
                Step 3: Reject unsafe or weak clusters
                Step 4: Select the BEST remaining cluster

                ====================
                OUTPUT FORMAT (STRICT JSON ONLY)
                ====================
                {{
                "group": ["word1", "word2", "word3"]
                }}
                """
        return prompt
    def generate_clue(group_words, enemy_words, neutral_words, assassin_words, board_state, shot = 0, banned_clues = []):
        if shot == 0:
            prompt = f"""
                You are a STRICT Codenames Spymaster.

                Your job is to convert a semantic group into a valid clue.

                ====================
                TASK
                ====================
                Given a GROUP of words, generate ONE valid clue word and its number.

                The clue must connect ALL words in the group.

                ====================
                INPUT
                ====================
                GROUP WORDS:
                {', '.join(group_words)}

                DANGEROUS WORDS (MUST AVOID):
                {', '.join(enemy_words + neutral_words + assassin_words)}

                BANNED CLUES (DO NOT REUSE):
                {', '.join(banned_clues)}

                ALL BOARD WORDS:
                {', '.join([item['word'] for item in board_state if not item['revealed']])}

                ====================
                STRICT RULES (MUST FOLLOW)
                ====================
                1. Output EXACTLY: <one_word> <number>
                - Example: pet 3
                - NO quotes, NO punctuation, NO extra text

                2. Clue must be ONE WORD only:
                - No spaces
                - No hyphens
                - No symbols
                - Letters only (a-z)

                3. Clue MUST NOT:
                - Be equal to ANY board word
                - Be a substring of ANY board word
                - Contain ANY board word as substring

                4. Clue MUST strongly relate to ALL group words

                5. Clue MUST NOT relate to dangerous words:
                - If it could plausibly connect → REJECT it

                6. Prefer:
                - Specific, high-signal concepts
                - Avoid generic clues (thing, object, item, etc.)

                ====================
                PROCESS (FOLLOW STEP BY STEP)
                ====================
                Step 1: Understand semantic relationship of group words
                Step 2: Generate multiple candidate clues mentally
                Step 3: Validate each clue against ALL rules
                Step 4: Select the safest + most informative valid clue

                ====================
                FINAL OUTPUT (ONLY THIS FORMAT)
                ====================
                clue number
                """
        elif shot == 1:
            prompt = f"""
                You are a STRICT Codenames Spymaster.

                Your job is to convert a semantic group into a valid clue.

                ====================
                TASK
                ====================
                Given a GROUP of words, generate ONE valid clue word and its number.

                The clue must connect ALL words in the group.

                ====================
                INPUT
                ====================
                GROUP WORDS:
                {', '.join(group_words)}

                DANGEROUS WORDS (MUST AVOID):
                {', '.join(enemy_words + neutral_words + assassin_words)}
                
                BANNED CLUES (DO NOT REUSE):
                {', '.join(banned_clues)}

                ALL BOARD WORDS:
                {', '.join([item['word'] for item in board_state if not item['revealed']])}

                ====================
                STRICT RULES (MUST FOLLOW)
                ====================
                1. Output EXACTLY: <one_word> <number>
                - Example: pet 3
                - NO quotes, NO punctuation, NO extra text

                2. Clue must be ONE WORD only:
                - No spaces
                - No hyphens
                - No symbols
                - Letters only (a-z)

                3. Clue MUST NOT:
                - Be equal to ANY board word
                - Be a substring of ANY board word
                - Contain ANY board word as substring

                4. Clue MUST strongly relate to ALL group words

                5. Clue MUST NOT relate to dangerous words:
                - If it could plausibly connect → REJECT it

                6. The number MUST equal the number of GROUP WORDS

                7. Prefer:
                - Specific, high-signal concepts
                - Avoid generic clues (thing, object, item, etc.)

                ====================
                GOOD EXAMPLES
                ====================

                Example 1:
                GROUP: cat, dog, hamster  
                DANGEROUS: car, tree  

                Reasoning:
                - All are pets → strong shared concept
                - "pet" does NOT relate to dangerous words

                Output:
                pet 3


                Example 2:
                GROUP: piano, guitar, violin  
                DANGEROUS: gun, bomb  

                Reasoning:
                - All are musical instruments
                - "music" safely connects all

                Output:
                music 3


                Example 3:
                GROUP: helicopter, airplane  
                DANGEROUS: bird  

                Reasoning:
                - Both are aircraft
                - Avoid "fly" (could relate to bird)
                - Use "aircraft" instead

                Output:
                aircraft 2


                ====================
                BAD EXAMPLES (DO NOT DO THIS)
                ====================

                Example 4:
                GROUP: cat, dog, hamster  
                Output:
                animal 3  
                ❌ BAD: too broad → may connect to dangerous words


                Example 5:
                GROUP: piano, guitar  
                Output:
                piano 2  
                ❌ BAD: clue is a board word


                Example 6:
                GROUP: helicopter  
                Output:
                vehicle 5  
                ❌ BAD: number does not match group size


                Example 7:
                GROUP: whale  
                DANGEROUS: ocean  

                Output:
                ocean 1  
                ❌ BAD: directly relates to dangerous word


                ====================
                PROCESS (FOLLOW STEP BY STEP)
                ====================
                Step 1: Understand semantic relationship of group words  
                Step 2: Generate multiple candidate clues mentally  
                Step 3: Validate each clue against ALL rules  
                Step 4: Select the safest + most informative valid clue  

                ====================
                FINAL OUTPUT (ONLY THIS FORMAT)
                ====================
                clue number
                """
        return prompt


def operative_prompt(num_guesses, clue_word, available_words):
    prompt = f"""
        You are playing Codenames as the Operative.

        TASK:
        Select exactly {num_guesses} words from AVAILABLE list that best match the clue.

        CLUE:
        {clue_word}

        

        AVAILABLE WORDS:
        {', '.join(available_words)}

        RULES:
        - Choose ONLY from AVAILABLE WORDS
        - No new words allowed
        - No combining words
        - Output must be exact matches

        OUTPUT FORMAT:
        word1, word2, word3
        """
    return prompt