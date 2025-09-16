import { useEffect, useState } from 'react';

export const useWebSocket = (url: string) => {
  const [data, setData] = useState<any>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(url);
    socket.onmessage = (event) => setData(JSON.parse(event.data));
    socket.onclose = () => console.log('WebSocket closed');
    socket.onerror = (error) => console.error('WebSocket error', error);
    setWs(socket);

    return () => socket.close();
  }, [url]);

  return { data, ws };
};
