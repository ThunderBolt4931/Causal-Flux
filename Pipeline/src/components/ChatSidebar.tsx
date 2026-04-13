// import { Plus, MessageSquare, Trash2 } from "lucide-react";
// import { Chat } from "@/types/chat";
// import { cn } from "@/lib/utils";

// interface ChatSidebarProps {
//   chats: Chat[];
//   activeChatId: string | null;
//   onNewChat: () => void;
//   onSelectChat: (chatId: string) => void;
//   onDeleteChat: (chatId: string) => void;
// }

// export function ChatSidebar({
//   chats,
//   activeChatId,
//   onNewChat,
//   onSelectChat,
//   onDeleteChat,
// }: ChatSidebarProps) {
//   return (
//     <aside className="flex h-screen w-16 flex-col bg-sidebar border-r border-sidebar-border transition-all duration-300 data-[expanded=true]:w-64 group"
//       data-expanded={chats.length > 0}
//     >
//       {/* New Chat Button */}
//       <div className="flex items-center justify-center p-4 border-b border-sidebar-border">
//         <button
//           onClick={onNewChat}
//           className="flex h-10 w-10 items-center justify-center rounded-lg bg-sidebar-accent hover:bg-primary/20 transition-all duration-200 hover:scale-105 group-data-[expanded=true]:w-full group-data-[expanded=true]:justify-start group-data-[expanded=true]:gap-3 group-data-[expanded=true]:px-4"
//           title="New Chat"
//         >
//           <Plus className="h-5 w-5 text-primary" />
//           <span className="hidden text-sm font-medium text-sidebar-foreground group-data-[expanded=true]:inline">
//             New Chat
//           </span>
//         </button>
//       </div>

//       {/* Chat History */}
//       <div className="flex-1 overflow-y-auto scrollbar-thin py-2">
//         {chats.length > 0 && (
//           <div className="px-2 space-y-1">
//             {chats.map((chat, index) => (
//               <button
//                 key={chat.id}
//                 onClick={() => onSelectChat(chat.id)}
//                 className={cn(
//                   "group/item flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-all duration-200 animate-slide-in",
//                   activeChatId === chat.id
//                     ? "bg-sidebar-accent text-sidebar-accent-foreground"
//                     : "text-sidebar-foreground hover:bg-sidebar-accent/50"
//                 )}
//                 style={{ animationDelay: `${index * 50}ms` }}
//               >
//                 <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
//                 <span className="hidden truncate text-sm group-data-[expanded=true]:inline">
//                   {chat.title}
//                 </span>
//                 <button
//                   onClick={(e) => {
//                     e.stopPropagation();
//                     onDeleteChat(chat.id);
//                   }}
//                   className="ml-auto hidden h-6 w-6 items-center justify-center rounded opacity-0 hover:bg-destructive/20 group-hover/item:opacity-100 group-data-[expanded=true]:flex transition-opacity"
//                   title="Delete chat"
//                 >
//                   <Trash2 className="h-3.5 w-3.5 text-destructive" />
//                 </button>
//               </button>
//             ))}
//           </div>
//         )}
//       </div>
//     </aside>
//   );
// }

import { useState } from "react";
import {
  Plus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { Chat } from "@/types/chat";
import { cn } from "@/lib/utils";

interface ChatSidebarProps {
  chats: Chat[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
}

export function ChatSidebar({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: ChatSidebarProps) {
  const [expanded, setExpanded] = useState(true);

  return (
    <aside
      className={cn(
        "flex h-screen flex-col bg-sidebar border-r border-[hsl(var(--primary)/0.5)] transition-all duration-300 group",
        expanded ? "w-64" : "w-16"
      )}
      data-expanded={expanded}
    >
      {/* Toggle button */}
      <div className="flex justify-end p-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="h-8 w-8 flex items-center justify-center rounded hover:bg-sidebar-accent/40"
          title="Toggle sidebar"
        >
          {expanded ? (
            <PanelLeftClose className="h-5 w-5 text-primary" />
          ) : (
            <PanelLeftOpen className="h-5 w-5 text-primary" />
          )}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="flex items-center justify-center p-4 border-b border-[hsl(var(--primary)/0.5)]">
        <button
          onClick={onNewChat}
          className={cn(
            "flex h-10 items-center justify-center rounded-lg bg-sidebar-accent hover:bg-primary/20 transition-all duration-200 hover:scale-105",
            expanded ? "w-full gap-3 px-4 justify-start" : "w-10"
          )}
        >
          <Plus className="h-5 w-5 text-primary" />
          {expanded && (
            <span className="text-sm font-medium text-sidebar-foreground">
              New Chat
            </span>
          )}
        </button>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin py-2">
        {chats.length > 0 && (
          <div className="px-2 space-y-1">
            {chats.map((chat, index) => (
              <div
                key={chat.id}
                className="relative animate-slide-in"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <button
                  onClick={() => onSelectChat(chat.id)}
                  className={cn(
                    "group/item flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-all duration-200",
                    activeChatId === chat.id
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent/50"
                  )}
                >
                  <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />

                  {expanded && (
                    <span className="truncate text-sm">{chat.title}</span>
                  )}

                  {/* Delete button */}
                  {expanded && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChat(chat.id);
                      }}
                      className="ml-auto h-6 w-6 flex items-center justify-center rounded opacity-0 hover:bg-destructive/20 group-hover/item:opacity-100 transition-opacity"
                      title="Delete chat"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-destructive" />
                    </button>
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
