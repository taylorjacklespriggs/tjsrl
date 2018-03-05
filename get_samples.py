def get_samples(game, keyboard_input):
    keyboard_input.add_to_window(game.env.unwrapped.viewer.window)
    keyboard_actions = keyboard_input.get_action_loop()
    while True:
        game.render()
        action = next(keyboard_actions)
        state, reward, done = game.do_action(action)
        yield action, state, reward
        if keyboard_input.finished:
            return
        if done:
            game.reset()

if __name__ == '__main__':
    from ll import LunarLander
    from keyboard_input import KeyboardInput
    print(game.reset())
    for action, state, reward in get_samples(LunarLander(), KeyboardInput()):
        print(state, reward)
