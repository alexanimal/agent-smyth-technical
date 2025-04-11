# Chat Application

A modern, feature-rich chat application with a clean UI and enhanced user experience.

## Features

- **Real-time Chat**: Send messages and receive responses with AI
- **Streaming Responses**: Watch AI responses come in real-time
- **Message Persistence**: Chat messages persist across browser refreshes using localStorage
- **Rich Markdown Support**: Full markdown rendering including code blocks with syntax highlighting
- **Code Block Management**: Copy code blocks to clipboard with a single click
- **Responsive Design**: Works well on all screen sizes
- **Custom UI Components**: Beautifully designed avatars, message containers, and more
- **Visual Feedback**: Loading indicators and animations enhance the user experience
- **Expandable Messages**: AI responses can expand up to 80% of the container width
- **Justified Text**: Improved readability with properly justified text in AI responses
- **Settings Panel**: Configure the behavior of the chat with the settings panel
- **Dark Mode Support**: Native support for light and dark themes

## Technical Features

- TypeScript for type safety
- React Hooks for state management
- Zustand for global state with localStorage persistence
- TailwindCSS for styling
- React Markdown for message rendering
- Syntax highlighting for code blocks
- Custom hooks for localStorage, clipboard management

## Getting Started

### Prerequisites

- Node.js 16+ and npm

### Installation

1. Clone the repository
2. Navigate to the UI directory:

```bash
cd ui
```

3. Install dependencies:

```bash
npm install
```

4. Start the development server:

```bash
npm run dev
```

5. Open the application in your browser at http://localhost:3000

## Building for Production

```bash
npm run build
```

## Configuration

- `API_BASE_URL` in `api/chatAPI.ts` - Set to the base URL of your backend API
- Model settings in the settings panel - Adjust based on your available models

## Implementation Details

### Message Persistence

Messages are automatically saved to localStorage and restored when the application loads. This is implemented using Zustand's persist middleware. The chat state is serialized to localStorage, allowing for a seamless experience across page refreshes.

### AI Message Display

AI response messages can expand up to 80% of the viewport width, allowing for better display of code blocks, tables, and other rich content. The implementation uses responsive design principles to ensure proper display across different screen sizes.

### Text Justification

AI response messages use justified text alignment for improved readability of longer paragraphs. This is applied using Tailwind's `text-justify` utility class along with additional CSS rules in the global stylesheet for consistent rendering across different types of content.

### Code Copy Functionality

Code blocks include a copy button that allows users to copy the code to their clipboard with a single click. This is implemented using the Clipboard API with a fallback for older browsers.
