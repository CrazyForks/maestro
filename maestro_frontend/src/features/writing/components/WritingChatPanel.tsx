import React, { useState, useEffect, useRef } from 'react'
import { Button } from '../../../components/ui/button'
import { Card, CardContent } from '../../../components/ui/card'
import { useToast } from '../../../components/ui/toast'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../../../components/ui/select'
import { Send, Loader2, Bot, Trash2, Sparkles, Settings, SlidersHorizontal } from 'lucide-react'
import { CustomSystemPromptModal } from './CustomSystemPromptModal'
import { WritingSearchSettingsModal } from './WritingSearchSettingsModal'
import { apiClient } from '../../../config/api'
import { useWritingStore } from '../store'
import { getDocumentGroups } from '../../documents/api'
import type { DocumentGroup } from '../../documents/types'
import { MessageBubble } from './MessageBubble'
import { useTheme } from '../../../contexts/ThemeContext'
import { useSettingsStore } from '../../auth/components/SettingsStore'
// import { ensureDate } from '../../../utils/timezone'

export const WritingChatPanel: React.FC = () => {
  const [message, setMessage] = useState('')
  const [documentGroups, setDocumentGroups] = useState<DocumentGroup[]>([])
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null)
  const [showCustomPromptModal, setShowCustomPromptModal] = useState(false)
  const [showSearchSettingsModal, setShowSearchSettingsModal] = useState(false)
  
  // Get user settings for defaults
  const { settings: userSettings } = useSettingsStore()
  const userParams = userSettings?.research_parameters
  
  // Single source of truth for all search settings - initialized with user preferences
  const [searchSettings, setSearchSettings] = useState({
    useWebSearch: false,  // Default to off
    deepSearch: false,
    maxIterations: userParams?.writing_search_max_iterations ?? 1,
    maxQueries: userParams?.writing_search_max_queries ?? 3,
    deepSearchIterations: userParams?.writing_deep_search_iterations ?? 3,
    deepSearchQueries: userParams?.writing_deep_search_queries ?? 5,
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { addToast } = useToast()

  // Get state and actions from writing store
  const {
    getCurrentMessages,
    getSessionLoading,
    currentSession,
    activeChat,
    createChat,
    setActiveChat,
    clearMessages,
    sendMessage,
    regenerateMessage,
    removeMessage,
    getCurrentAgentStatus,
  } = useWritingStore()
  
  const agentStatus = getCurrentAgentStatus()
  
  // Get current messages using the new pattern
  const messages = getCurrentMessages()
  
  // Get loading state - check session loading first, then fallback to chat loading
  // Also consider agent status - if agent is working, we should show loading
  const sessionLoading = currentSession 
    ? getSessionLoading(currentSession.id) 
    : (activeChat ? getSessionLoading(activeChat.id) : false) || getSessionLoading('session_creating')
  
  const agentWorking = agentStatus && agentStatus !== '' && !['idle', 'complete', 'error'].includes(agentStatus)
  const isLoading = sessionLoading || !!agentWorking

  // Auto-scroll to bottom when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])
  

  // Listen for chat title updates from WebSocket
  // Note: This is now handled by WritingChatSidebar to avoid duplicate handlers
  // The store already updates the activeChat title via WebSocket

  // Load document groups on component mount
  useEffect(() => {
    const fetchGroups = async () => {
      try {
        const groups = await getDocumentGroups()
        setDocumentGroups(groups)
      } catch (error) {
        console.error('Failed to fetch document groups:', error)
      }
    }
    fetchGroups()
  }, []) // Only load once on mount
  
  // Track if we're loading settings from a chat to prevent saving during load
  const [isLoadingChatSettings, setIsLoadingChatSettings] = useState(false)
  
  // Load saved tool settings when activeChat changes
  useEffect(() => {
    if (activeChat) {
      setIsLoadingChatSettings(true)
      
      if (activeChat.settings?.tools) {
        const tools = activeChat.settings.tools
        setSelectedGroupId(tools.documentGroupId || null)
        setSearchSettings({
          useWebSearch: tools.useWebSearch ?? false,
          deepSearch: tools.deepSearch ?? false,
          maxIterations: tools.maxIterations ?? userParams?.writing_search_max_iterations ?? 1,
          maxQueries: tools.maxQueries ?? userParams?.writing_search_max_queries ?? 3,
          deepSearchIterations: tools.deepSearchIterations ?? userParams?.writing_deep_search_iterations ?? 3,
          deepSearchQueries: tools.deepSearchQueries ?? userParams?.writing_deep_search_queries ?? 5,
        })
      } else {
        // No saved settings, use defaults
        setSelectedGroupId(null)
        setSearchSettings({
          useWebSearch: false,
          deepSearch: false,
          maxIterations: userParams?.writing_search_max_iterations ?? 1,
          maxQueries: userParams?.writing_search_max_queries ?? 3,
          deepSearchIterations: userParams?.writing_deep_search_iterations ?? 3,
          deepSearchQueries: userParams?.writing_deep_search_queries ?? 5,
        })
      }
      
      // Allow saving after a short delay
      setTimeout(() => {
        setIsLoadingChatSettings(false)
      }, 100)
    }
  }, [activeChat?.id]) // Only trigger when chat ID changes
  
  // Save tool settings when they change (but not during initial load)
  const saveToolSettings = async () => {
    if (!activeChat || isLoadingChatSettings) return
    
    try {
      const chatSettings = {
        ...(activeChat.settings || {}),
        tools: {
          documentGroupId: selectedGroupId,
          useWebSearch: searchSettings.useWebSearch,
          deepSearch: searchSettings.deepSearch,
          maxIterations: searchSettings.maxIterations,
          maxQueries: searchSettings.maxQueries,
          deepSearchIterations: searchSettings.deepSearchIterations,
          deepSearchQueries: searchSettings.deepSearchQueries,
        }
      }
      
      // Update chat settings via API
      await apiClient.put(`/api/chats/${activeChat.id}`, {
        settings: chatSettings
      })
      
      // Update local state
      useWritingStore.setState((state) => ({
        activeChat: { ...activeChat, settings: chatSettings }
      }))
    } catch (error) {
      console.error('Failed to save tool settings:', error)
    }
  }
  
  // Debounce saving tool settings
  useEffect(() => {
    if (!activeChat || isLoadingChatSettings) return
    
    const timer = setTimeout(() => {
      saveToolSettings()
    }, 500) // Save after 500ms of no changes
    
    return () => clearTimeout(timer)
  }, [selectedGroupId, searchSettings]) // Don't include activeChat.id to avoid saving during chat switch

  const handleSendMessage = async () => {
    if (!message.trim() || isLoading) return

    const userMessage = message.trim()
    setMessage('')

    try {
      // Use the store's sendMessage method which handles adding the user message and getting the response
      // The sendMessage method will create a chat/session if none exists
      await sendMessage(userMessage, {
        documentGroupId: selectedGroupId,
        useWebSearch: searchSettings.useWebSearch,
        deepSearch: searchSettings.deepSearch,
        maxIterations: searchSettings.deepSearch ? searchSettings.deepSearchIterations : searchSettings.maxIterations,
        maxQueries: searchSettings.deepSearch ? searchSettings.deepSearchQueries : searchSettings.maxQueries
      })
      
    } catch (error) {
      console.error('Error sending writing message:', error)
      addToast({
        type: 'error',
        title: 'Chat Error',
        message: 'Failed to send message. Please try again.',
        duration: 5000
      })
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleRegenerate = async (messageId: string) => {
    try {
      await regenerateMessage(messageId, {
        documentGroupId: selectedGroupId,
        useWebSearch: searchSettings.useWebSearch,
        deepSearch: searchSettings.deepSearch,
        maxIterations: searchSettings.deepSearch ? searchSettings.deepSearchIterations : searchSettings.maxIterations,
        maxQueries: searchSettings.deepSearch ? searchSettings.deepSearchQueries : searchSettings.maxQueries
      });
    } catch (error) {
      console.error('Error regenerating message:', error)
      addToast({
        type: 'error',
        title: 'Regeneration Error',
        message: 'Failed to regenerate response. Please try again.',
        duration: 5000
      })
    }
  }

  const getAgentStatusMessage = (status: string): string => {
    // The status now comes as a descriptive string from the backend
    // If it starts with certain keywords, we can still handle them specially
    if (!status || status === 'idle') {
      return ''
    }
    
    // Return the status message as-is since backend now sends descriptive messages
    return status
  }

  // Show welcome screen if no active chat is selected
  if (!activeChat) {
    const { theme } = useTheme()
    
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="text-center max-w-md mx-auto p-8">
          <div className="flex justify-center mb-6">
            <img 
              src={theme === 'dark' ? '/icon_dark.png' : '/icon_original.png'} 
              alt="MAESTRO Logo" 
              className="h-16 w-16 transition-transform hover:scale-105"
            />
          </div>
          <h2 className="text-2xl font-bold text-foreground mb-4">
            Welcome to MAESTRO Writing
          </h2>
          <p className="text-muted-foreground mb-6">
            AI writing assistant ready to help with content generation and editing.
          </p>
          <Button 
            onClick={async () => {
              try {
                const newChat = await createChat()
                await setActiveChat(newChat.id)
              } catch (error) {
                console.error('Failed to create new chat:', error)
                addToast({
                  type: 'error',
                  title: 'Error',
                  message: 'Failed to create new chat. Please try again.',
                  duration: 5000
                })
              }
            }}
            className="inline-flex items-center"
          >
            <Sparkles className="h-4 w-4 mr-2" />
            Start New Chat
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Chat Header */}
      <div className="bg-header-background border-b border-border px-4 py-4 min-h-[88px] flex items-center">
        <div className="flex flex-col w-full">
          {/* Title */}
          <h3 className="text-base font-semibold text-text-primary">
            {activeChat?.title || 'Writing Chat'}
          </h3>
          
          {/* Message count and buttons on same line */}
          <div className="flex items-center justify-between mt-1">
            <p className="text-xs text-text-secondary">
              {messages.length === 0
                ? 'Start the conversation'
                : `${messages.length} messages`}
            </p>
            
            {/* Action buttons */}
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSearchSettingsModal(true)}
                className="h-6 px-2 text-xs"
              >
                <SlidersHorizontal className="h-3 w-3 mr-1" />
                Search Settings
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowCustomPromptModal(true)}
                className="h-6 px-2 text-xs"
              >
                <Settings className="h-3 w-3 mr-1" />
                Custom Prompt
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (window.confirm('Are you sure you want to clear the chat?')) {
                    clearMessages();
                  }
                }}
                disabled={messages.length === 0}
                className="h-6 px-2 text-xs"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-3">
                <Bot className="h-5 w-5 text-primary" />
              </div>
              <h3 className="text-base font-medium text-foreground mb-2">
                Ready to help with your writing!
              </h3>
              <p className="text-muted-foreground text-xs max-w-sm mx-auto">
                Ask for content generation, editing assistance, or document structure help.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onRegenerate={() => handleRegenerate(msg.id)}
                onDelete={async () => {
                  try {
                    await removeMessage(msg.id)
                  } catch (error) {
                    console.error('Failed to delete message:', error)
                    addToast({
                      type: 'error',
                      title: 'Delete Failed',
                      message: 'Failed to delete message. Please try again.',
                      duration: 5000
                    })
                  }
                }}
                isRegenerating={isLoading}
              />
            ))
          )}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="flex items-start space-x-2">
                <div className="flex-shrink-0 w-7 h-7 rounded-full bg-secondary flex items-center justify-center">
                  <Bot className="h-3.5 w-3.5 text-secondary-foreground" />
                </div>
                <div className="bg-card border border-border text-foreground max-w-xs lg:max-w-2xl px-3 py-2 rounded-xl rounded-bl-md shadow-sm">
                  <div className="flex items-center space-x-2">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                    <span className="text-xs text-muted-foreground">
                      {getAgentStatusMessage(agentStatus)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Message Input */}
      <div className="bg-card border-t border-border p-4">
        <div className="max-w-4xl mx-auto">
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-col">
                <div className="flex space-x-4">
                  <textarea
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask for writing help, request content generation, or give structural commands..."
                    className="flex-1 resize-none border-0 focus:ring-0 focus:outline-none text-xs min-h-[18px] max-h-28 bg-transparent"
                    rows={1}
                    disabled={isLoading}
                    style={{
                      height: 'auto',
                      minHeight: '20px'
                    }}
                    onInput={(e) => {
                      const target = e.target as HTMLTextAreaElement
                      target.style.height = 'auto'
                      target.style.height = target.scrollHeight + 'px'
                    }}
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={!message.trim() || isLoading}
                    size="sm"
                    className="self-end"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <div className="mt-2 space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="flex items-center space-x-1.5 min-w-0">
                      <label className="text-xs text-muted-foreground whitespace-nowrap">Document Group:</label>
                      <Select
                        value={selectedGroupId || 'none'}
                        onValueChange={(value) => {
                          if (value === "none") {
                            setSelectedGroupId(null);
                          } else {
                            setSelectedGroupId(value);
                          }
                        }}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="text-xs h-6 w-[110px]">
                          <SelectValue placeholder="None" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          {documentGroups.map((group) => (
                            <SelectItem key={group.id} value={group.id}>
                              {group.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="flex items-center space-x-1.5">
                      <label className="text-xs text-muted-foreground whitespace-nowrap">Web Search:</label>
                      <button
                        type="button"
                        onClick={() => {
                          setSearchSettings(prev => ({ ...prev, useWebSearch: !prev.useWebSearch }))
                        }}
                        disabled={isLoading}
                        className={`relative inline-flex h-3.5 w-6 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
                          searchSettings.useWebSearch ? 'bg-primary' : 'bg-secondary'
                        } ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                      >
                        <span
                          className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                            searchSettings.useWebSearch ? 'translate-x-3' : 'translate-x-0.5'
                          }`}
                        />
                      </button>
                      <span className={`text-xs ${searchSettings.useWebSearch ? 'text-primary' : 'text-muted-foreground'}`}>
                        {searchSettings.useWebSearch ? 'On' : 'Off'}
                      </span>
                    </div>
                    
                    {/* Deep Search Toggle - Show when either web search is on OR documents are selected */}
                    {(searchSettings.useWebSearch || selectedGroupId) && (
                      <div className="flex items-center space-x-1.5">
                        <label className="text-xs text-muted-foreground whitespace-nowrap">Deep Search:</label>
                        <button
                          type="button"
                          onClick={() => {
                            setSearchSettings(prev => ({ ...prev, deepSearch: !prev.deepSearch }))
                          }}
                          disabled={isLoading || (!searchSettings.useWebSearch && !selectedGroupId)}
                          className={`relative inline-flex h-3.5 w-6 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
                            searchSettings.deepSearch ? 'bg-primary' : 'bg-secondary'
                          } ${isLoading || (!searchSettings.useWebSearch && !selectedGroupId) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                        >
                          <span
                            className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                              searchSettings.deepSearch ? 'translate-x-3' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                        <span className={`text-xs ${searchSettings.deepSearch ? 'text-primary' : 'text-muted-foreground'}`}>
                          {searchSettings.deepSearch ? 'On' : 'Off'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <p className="text-xs text-muted-foreground mt-1.5 text-center">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* Custom System Prompt Modal */}
      <CustomSystemPromptModal
        isOpen={showCustomPromptModal}
        onClose={() => setShowCustomPromptModal(false)}
      />
      
      {/* Search Settings Modal */}
      <WritingSearchSettingsModal
        isOpen={showSearchSettingsModal}
        onClose={() => setShowSearchSettingsModal(false)}
        settings={searchSettings}
        onSave={(newSettings) => {
          setSearchSettings(newSettings)
        }}
      />
    </div>
  )
}
