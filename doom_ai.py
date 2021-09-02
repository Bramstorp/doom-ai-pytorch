import vizdoom as vzd

def doom_game_initializer():
    game = vzd.DoomGame()
    game.load_config("scenarios/simpler_basic.cfg")
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized")

    return game