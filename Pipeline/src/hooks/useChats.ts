import { useState, useCallback, useEffect } from "react";
import { Chat, Message } from "@/types/chat";
import { supabase } from "@/integrations/supabase/client";
import { API_BASE_URL, apiUrl } from "@/lib/api";

export function useChats() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const activeChat = chats.find((c) => c.id === activeChatId) || null;

  // Load chats from database on mount
  useEffect(() => {
    const loadChats = async () => {
      const { data: chatsData, error: chatsError } = await supabase
        .from("chats")
        .select("*")
        .order("updated_at", { ascending: false });

      if (chatsError) {
        console.error("Error loading chats:", chatsError);
        return;
      }

      if (!chatsData || chatsData.length === 0) {
        setChats([]);
        return;
      }

      // Load messages for all chats
      const { data: messagesData, error: messagesError } = await supabase
        .from("messages")
        .select("*")
        .in("chat_id", chatsData.map((c) => c.id))
        .order("created_at", { ascending: true });

      if (messagesError) {
        console.error("Error loading messages:", messagesError);
      }

      const loadedChats: Chat[] = chatsData.map((chat) => ({
        id: chat.id,
        title: chat.title,
        messages: (messagesData || [])
          .filter((m) => m.chat_id === chat.id)
          .map((m) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
            timestamp: new Date(m.created_at),
            // Try to load full metadata if available, otherwise construct partial
            metadata: m.metadata || (m.model ? { model: m.model, token_count: m.token_count } : undefined),
          })),
        createdAt: new Date(chat.created_at),
        updatedAt: new Date(chat.updated_at),
      }));

      setChats(loadedChats);
    };

    loadChats();
  }, []);

  const createNewChat = useCallback(async () => {
    const { data, error } = await supabase
      .from("chats")
      .insert({ title: "New Chat" })
      .select()
      .single();

    if (error || !data) {
      console.error("Error creating chat:", error);
      return null;
    }

    const newChat: Chat = {
      id: data.id,
      title: data.title,
      messages: [],
      createdAt: new Date(data.created_at),
      updatedAt: new Date(data.updated_at),
    };

    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    return newChat.id;
  }, []);

  const deleteChat = useCallback(async (chatId: string) => {
    const { error } = await supabase.from("chats").delete().eq("id", chatId);

    if (error) {
      console.error("Error deleting chat:", error);
      return;
    }

    setChats((prev) => prev.filter((c) => c.id !== chatId));
    if (activeChatId === chatId) {
      setActiveChatId(null);
    }
  }, [activeChatId]);

  const sendMessage = useCallback(
    async (message: string, modelChoice: string, taskMode: "task1" | "task2") => {
      let chatId = activeChatId;

      // Create new chat if none active
      if (!chatId) {
        const title = message.slice(0, 40) + (message.length > 40 ? "..." : "");
        const { data, error } = await supabase
          .from("chats")
          .insert({ title })
          .select()
          .single();

        if (error || !data) {
          console.error("Error creating chat:", error);
          return;
        }

        const newChat: Chat = {
          id: data.id,
          title: data.title,
          messages: [],
          createdAt: new Date(data.created_at),
          updatedAt: new Date(data.updated_at),
        };

        setChats((prev) => [newChat, ...prev]);
        chatId = newChat.id;
        setActiveChatId(chatId);
      }

      // Add user message to DB
      const { data: userMsgData, error: userMsgError } = await supabase
        .from("messages")
        .insert({
          chat_id: chatId,
          role: "user",
          content: message,
        })
        .select()
        .single();

      if (userMsgError || !userMsgData) {
        console.error("Error saving user message:", userMsgError);
        return;
      }

      const userMessage: Message = {
        id: userMsgData.id,
        role: "user",
        content: message,
        timestamp: new Date(userMsgData.created_at),
      };

      // Update title if first message
      const currentChat = chats.find((c) => c.id === chatId);
      if (currentChat && currentChat.messages.length === 0) {
        const newTitle = message.slice(0, 40) + (message.length > 40 ? "..." : "");
        await supabase.from("chats").update({ title: newTitle }).eq("id", chatId);
        setChats((prev) =>
          prev.map((c) =>
            c.id === chatId
              ? { ...c, title: newTitle, messages: [...c.messages, userMessage], updatedAt: new Date() }
              : c
          )
        );
      } else {
        setChats((prev) =>
          prev.map((c) =>
            c.id === chatId
              ? { ...c, messages: [...c.messages, userMessage], updatedAt: new Date() }
              : c
          )
        );
      }

      setIsLoading(true);

      try {
        const response = await fetch(apiUrl("/chat"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            model_choice: modelChoice,
            task_mode: taskMode,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        // Clean up the response
        let cleanResponse = data.response;
        const footerIndex = cleanResponse.lastIndexOf("\n\n---\n");
        if (footerIndex !== -1) {
          cleanResponse = cleanResponse.slice(0, footerIndex);
        }

        // Save assistant message to DB
        const { data: assistantMsgData, error: assistantMsgError } = await supabase
          .from("messages")
          .insert({
            chat_id: chatId,
            role: "assistant",
            content: cleanResponse,
            model: data.metadata?.model,
            token_count: data.metadata?.token_count,
            metadata: data.metadata,
          })
          .select()
          .single();

        if (assistantMsgError) {
          console.warn("Could not save full metadata to DB (ignore if no metadata column):", assistantMsgError);
        }

        const msgId = assistantMsgData?.id || crypto.randomUUID();
        const msgDate = assistantMsgData ? new Date(assistantMsgData.created_at) : new Date();

        const assistantMessage: Message = {
          id: msgId,
          role: "assistant",
          content: cleanResponse,
          timestamp: msgDate,
          metadata: data.metadata,
        };

        setChats((prev) =>
          prev.map((c) =>
            c.id === chatId
              ? { ...c, messages: [...c.messages, assistantMessage], updatedAt: new Date() }
              : c
          )
        );
      } catch (error) {
        console.error("Failed to send message:", error);

        // Save error message to DB
        const errorContent = "Sorry, I couldn't connect to the backend. Please check the server status.";
        const { data: errorMsgData } = await supabase
          .from("messages")
          .insert({
            chat_id: chatId,
            role: "assistant",
            content: errorContent,
          })
          .select()
          .single();

        const errorMessage: Message = {
          id: errorMsgData?.id || crypto.randomUUID(),
          role: "assistant",
          content: errorContent,
          timestamp: new Date(),
        };

        setChats((prev) =>
          prev.map((c) =>
            c.id === chatId
              ? { ...c, messages: [...c.messages, errorMessage], updatedAt: new Date() }
              : c
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeChatId, chats]
  );

  return {
    chats,
    activeChat,
    activeChatId,
    isLoading,
    createNewChat,
    setActiveChatId,
    deleteChat,
    sendMessage,
  };
}