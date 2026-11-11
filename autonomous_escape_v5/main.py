"""Autonomous Escape v2.

  python main.py            3D pilot view (default)
  python main.py --2d       the original top-down 2D game
  python main.py --help     direct-start options (--mode, --camera, --play, ...)
"""
import sys

if __name__ == '__main__':
    if '--2d' in sys.argv:
        sys.argv.remove('--2d')
        from ui.app import main
    else:
        from ui3d.app3d import main
    main()
