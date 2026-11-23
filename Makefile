# Root targets. Paper: see paper/Makefile.
.PHONY: video

# 60-second video -> video/aftermarket_60s.mp4 and video/contact_sheet.png (needs ffmpeg on PATH)
video:
	python video/build.py
