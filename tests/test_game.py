import pytest
from src.engine.game import CodenamesGame

# A dummy vocabulary for testing
TEST_VOCAB = [f"Word{i}" for i in range(1, 51)]

def test_board_initialization():
    """Test that the board is created with exactly the right card distribution."""
    # Arrange & Act
    game = CodenamesGame(TEST_VOCAB)
    
    # Assert
    assert len(game.words) == 25
    assert len(game.identities) == 25
    assert game.identities.count('Red') == 9
    assert game.identities.count('Blue') == 8
    assert game.identities.count('Neutral') == 7
    assert game.identities.count('Assassin') == 1

def test_valid_and_invalid_clues():
    """Test that the Spymaster cannot give a clue that is on the board."""
    # Arrange
    game = CodenamesGame(TEST_VOCAB)
    visible_word = game.words[0]  # Grab a word that is currently on the board
    
    # Act & Assert
    # 1. Exact match should be invalid
    assert game.is_valid_clue(visible_word) == False
    
    # 2. Substring/Superstring should be invalid (e.g., 'Snow' invalidates 'Snowman')
    assert game.is_valid_clue(visible_word + "ing") == False 
    
    # 3. A completely different word should be valid
    assert game.is_valid_clue("CompletelyRandomWord") == True

def test_guessing_red_card_continues_turn():
    """Test that guessing your own team's card allows you to keep guessing."""
    # Arrange
    game = CodenamesGame(TEST_VOCAB)
    # Find the index of a 'Red' card
    red_index = game.identities.index('Red')
    red_word = game.words[red_index]
    
    # Act
    identity, continue_turn, game_over = game.process_guess(red_word)
    
    # Assert
    assert identity == 'Red'
    assert continue_turn == True
    assert game_over == False
    assert game.red_found == 1
    assert game.revealed[red_index] == True

def test_guessing_neutral_ends_turn():
    """Test that guessing a Neutral card ends the turn but not the game."""
    # Arrange
    game = CodenamesGame(TEST_VOCAB)
    neutral_idx = game.identities.index('Neutral')
    neutral_word = game.words[neutral_idx]
    
    # Act
    identity, continue_turn, game_over = game.process_guess(neutral_word)
    
    # Assert
    assert identity == 'Neutral'
    assert continue_turn == False
    assert game_over == False

def test_guessing_assassin_ends_game():
    """Test that hitting the Assassin triggers an immediate loss."""
    # Arrange
    game = CodenamesGame(TEST_VOCAB)
    assassin_idx = game.identities.index('Assassin')
    assassin_word = game.words[assassin_idx]
    
    # Act
    identity, continue_turn, game_over = game.process_guess(assassin_word)
    
    # Assert
    assert identity == 'Assassin'
    assert continue_turn == False
    assert game_over == True
    assert game.winner == 'Assassin'

def test_winning_the_game():
    """Test that finding all 9 Red cards triggers a win."""
    # Arrange
    game = CodenamesGame(TEST_VOCAB)
    red_words = [w for w, i in zip(game.words, game.identities) if i == 'Red']
    
    # Act: Guess the first 8 red words
    for word in red_words[:8]:
        game.process_guess(word)
        
    assert game.is_game_over == False # Game shouldn't be over yet
    
    # Act: Guess the 9th red word
    identity, continue_turn, game_over = game.process_guess(red_words[8])
    
    # Assert
    assert game_over == True
    assert game.winner == 'Red'