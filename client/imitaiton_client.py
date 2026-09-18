# -*- coding: utf-8 -*-
"""Legacy entry point for the audio-free imitation client."""

if __package__:
    from .run_imitation_client import main, run_imitation_client
else:
    from run_imitation_client import main, run_imitation_client


if __name__ == "__main__":
    main()
