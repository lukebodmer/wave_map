from wave_map.gui.trame_user_interface import TrameGui


def main():
    """Entry point for script usage"""
    app = TrameGui()
    app.server.start(port=1234, open_browser=True)


if __name__ == "__main__":
    main()
