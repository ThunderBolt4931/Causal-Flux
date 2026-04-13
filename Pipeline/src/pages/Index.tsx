import { ChatSidebar } from "@/components/ChatSidebar";
import { ChatContainer } from "@/components/ChatContainer";
import { useChats } from "@/hooks/useChats";
import FloatingLines from "@/components/FloatingLines";

const Index = () => {
  const {
    chats,
    activeChat,
    activeChatId,
    isLoading,
    createNewChat,
    setActiveChatId,
    deleteChat,
    sendMessage,
  } = useChats();

  const handleNewChat = () => {
    setActiveChatId(null);
  };

  return (
    <div className="relative h-screen w-full overflow-hidden bg-background">
      {/* FloatingLines Global Background */}
      <div className="absolute inset-0 z-0" style={{ width: '100%', height: '100%' }}>
        <FloatingLines
          enabledWaves={['top', 'middle', 'bottom']}
          lineCount={[10, 15, 20]}
          lineDistance={[8, 6, 4]}
          bendRadius={5.0}
          bendStrength={-0.5}
          interactive={true}
          parallax={true}
        />
      </div>

      {/* Content Wrapper */}
      <div className="relative z-10 flex h-full w-full pointer-events-none">

        {/* Sidebar - needs pointer events */}
        <div className="pointer-events-auto h-full">
          <ChatSidebar
            chats={chats}
            activeChatId={activeChatId}
            onNewChat={handleNewChat}
            onSelectChat={setActiveChatId}
            onDeleteChat={deleteChat}
          />
        </div>

        {/* Main Chat Area - needs pointer events only for interactive children */}
        <main className="flex-1 overflow-hidden pointer-events-auto">
          <ChatContainer
            messages={activeChat?.messages || []}
            onSend={sendMessage}
            isLoading={isLoading}
            isNewChat={!activeChatId}
          />
        </main>
      </div>
    </div>
  );
};

export default Index;
