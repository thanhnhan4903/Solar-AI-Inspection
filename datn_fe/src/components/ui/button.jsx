import React from "react";

export function Button({ className = "", children, onClick, ...props }) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
}
