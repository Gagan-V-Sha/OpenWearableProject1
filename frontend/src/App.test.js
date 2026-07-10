import { render, screen } from '@testing-library/react';
import App from './App';

test('renders loading state on first paint', () => {
  global.fetch = jest.fn(() => new Promise(() => {}));
  render(<App />);
  expect(screen.getByText(/loading your recovery data/i)).toBeInTheDocument();
});
