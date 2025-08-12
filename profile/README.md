# Editable Professional Profile (HTML)

This folder contains a clean, responsive profile page you can edit with your details. It's optimized for web and for printing to PDF.

## Files
- `index.html`: The profile page with placeholders like `[YOUR NAME]` and element IDs for easy editing.
- `styles.css`: The theme and layout using CSS variables.

## How to use
1. Open `index.html` in a browser.
2. Edit text directly in `index.html` (look for placeholders and IDs like `name`, `summary`, `projects`).
3. Update links in the contact section (`email`, `phone`, `portfolio`, `github`, `linkedin`).
4. Duplicate the project card block to add more projects.

## Customize the theme
- In `styles.css`, change the CSS variables at the top, for example:
  - `--primary` and `--accent` to set your brand colors.
  - Adjust background and text colors if you want a light theme.

## Add a profile photo
- Replace the `.avatar` element in `index.html` with an `<img src="your-photo.jpg" alt="[YOUR NAME]" class="avatar" />` and put your image file in the same folder.

## Export to PDF
- Click the "Download PDF" button or press `Ctrl/Cmd + P` and select "Save as PDF".
- Margins and colors are tuned for printing via the included print styles.

## Publish online (optional)
- You can host this on GitHub Pages, Netlify, or Vercel by deploying this folder.