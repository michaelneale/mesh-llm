import { useState } from "react";
import { AlertCircle, Bell, AlertTriangle } from "lucide-react";
import { Button, type ButtonProps } from "../../../../components/ui/button";
import { Input } from "../../../../components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "../../../../components/ui/dialog";
import { Badge } from "../../../../components/ui/badge";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "../../../../components/ui/toggle-group";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "../../../../components/ui/alert";

type ButtonVariant = NonNullable<ButtonProps["variant"]>;
type ButtonSize = NonNullable<ButtonProps["size"]>;

const buttonVariants: ButtonVariant[] = [
  "default",
  "secondary",
  "outline",
  "destructive",
];
const buttonSizes: ButtonSize[] = ["sm", "default", "lg"];

function PreviewCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <h3 className="font-medium text-sm">{title}</h3>
      {children}
    </div>
  );
}

function ButtonPreview() {
  const [variant, setVariant] = useState<ButtonVariant>("default");
  const [size, setSize] = useState<ButtonSize>("default");
  const [disabled, setDisabled] = useState(false);

  return (
    <PreviewCard title="Button">
      <div className="flex items-center gap-2">
        <Button variant={variant} size={size} disabled={disabled}>
          Preview
        </Button>
      </div>

      <div className="space-y-2">
        <span className="block text-xs text-muted-foreground font-medium">
          Variant
        </span>
        <ToggleGroup
          type="single"
          value={variant}
          onValueChange={(v: string) => v && setVariant(v as ButtonVariant)}
        >
          {buttonVariants.map((v) => (
            <ToggleGroupItem key={v} value={v} aria-label={v}>
              {v}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>

        <span className="block text-xs text-muted-foreground font-medium">
          Size
        </span>
        <ToggleGroup
          type="single"
          value={size}
          onValueChange={(v: string) => v && setSize(v as ButtonSize)}
        >
          {buttonSizes.map((s) => (
            <ToggleGroupItem key={s} value={s} aria-label={s}>
              {s}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>

        <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
          Disabled
          <input
            type="checkbox"
            checked={disabled}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setDisabled(e.target.checked)
            }
            className="h-4 w-4 rounded border-border accent-primary ml-auto"
          />
        </label>
      </div>
    </PreviewCard>
  );
}

function InputPreview() {
  const [placeholder, setPlaceholder] = useState("Enter some text…");
  const [type, setType] = useState<"text" | "email" | "password">("text");
  const [disabled, setDisabled] = useState(false);

  return (
    <PreviewCard title="Input">
      <Input type={type} placeholder={placeholder} disabled={disabled} />

      <div className="space-y-2">
        <span className="block text-xs text-muted-foreground font-medium">
          Type
        </span>
        <ToggleGroup
          type="single"
          value={type}
          onValueChange={(v: string) =>
            v && setType(v as "text" | "email" | "password")
          }
        >
          {(["text", "email", "password"] as const).map((t) => (
            <ToggleGroupItem key={t} value={t} aria-label={t}>
              {t}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      <div className="space-y-1.5">
        <label
          htmlFor="input-placeholder"
          className="text-xs text-muted-foreground font-medium"
        >
          Placeholder
        </label>
        <Input
          id="input-placeholder"
          value={placeholder}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setPlaceholder(e.target.value)
          }
          className="h-8 text-xs"
        />
      </div>

      <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
        <input
          type="checkbox"
          checked={disabled}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setDisabled(e.target.checked)
          }
          className="h-4 w-4 rounded border-border accent-primary"
        />
        Disabled
      </label>
    </PreviewCard>
  );
}

function DialogPreview() {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("Dialog Title");
  const [description, setDescription] = useState(
    "A short description of this dialog.",
  );

  return (
    <>
      <PreviewCard title="Dialog">
        <Button variant="outline" onClick={() => setOpen(true)}>
          Open Dialog
        </Button>

        <div className="space-y-2">
          <label
            htmlFor="dialog-title"
            className="text-xs text-muted-foreground font-medium"
          >
            Title
          </label>
          <Input
            id="dialog-title"
            value={title}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setTitle(e.target.value)
            }
            className="h-8 text-xs"
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="dialog-desc"
            className="text-xs text-muted-foreground font-medium"
          >
            Description
          </label>
          <Input
            id="dialog-desc"
            value={description}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setDescription(e.target.value)
            }
            className="h-8 text-xs"
          />
        </div>
      </PreviewCard>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <div className="space-y-1.5 pb-2">
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>{description}</DialogDescription>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => setOpen(false)}>Confirm</Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

type BadgeVariant = "default" | "secondary" | "outline";

const badgeVariantStyles: Record<BadgeVariant, string> = {
  default: "",
  secondary: "bg-secondary text-secondary-foreground",
  outline: "border-transparent bg-primary/10 text-primary",
};

function BadgePreview() {
  const [variant, setVariant] = useState<BadgeVariant>("default");
  const [label, setLabel] = useState("Badge");

  return (
    <PreviewCard title="Badge">
      <div className="flex items-center gap-2">
        <span
          className={`inline-flex items-center rounded-full border border-border/70 px-2.5 py-1 text-xs font-medium ${badgeVariantStyles[variant]}`}
        >
          {label || "(empty)"}
        </span>
      </div>

      <div className="space-y-2">
        <span className="block text-xs text-muted-foreground font-medium">
          Variant
        </span>
        <ToggleGroup
          type="single"
          value={variant}
          onValueChange={(v: string) => v && setVariant(v as BadgeVariant)}
        >
          {(["default", "secondary", "outline"] as const).map((v) => (
            <ToggleGroupItem key={v} value={v} aria-label={v}>
              {v}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>

        <span className="block text-xs text-muted-foreground font-medium">
          Label
        </span>
        <Input
          id="badge-label"
          value={label}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setLabel(e.target.value)
          }
          className="h-8 text-xs"
        />
      </div>
    </PreviewCard>
  );
}

function AlertPreview() {
  const [variant, setVariant] = useState<
    "default" | "primary" | "amber" | "destructive"
  >("primary");
  const [title, setTitle] = useState("Welcome to the public mesh");
  const [description, setDescription] = useState(
    "Mesh LLM is a project to let people contribute spare compute.",
  );

  return (
    <PreviewCard title="Alert / Banner">
      <Alert variant={variant}>
        {variant === "destructive" ? (
          <AlertCircle className="h-4 w-4" />
        ) : variant === "amber" ? (
          <AlertTriangle className="h-4 w-4" />
        ) : (
          <Bell className="h-4 w-4" />
        )}
        <AlertTitle>{title || "(empty)"}</AlertTitle>
        <AlertDescription>{description || "(no description)"}</AlertDescription>
      </Alert>

      <div className="space-y-2">
        <span className="block text-xs text-muted-foreground font-medium">
          Variant
        </span>
        <ToggleGroup
          type="single"
          value={variant}
          onValueChange={(v: string) =>
            v &&
            setVariant(v as "default" | "primary" | "amber" | "destructive")
          }
        >
          {(["primary", "amber", "destructive"] as const).map((v) => (
            <ToggleGroupItem key={v} value={v} aria-label={v}>
              {v}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>

        <div className="space-y-1.5">
          <label
            htmlFor="alert-title"
            className="text-xs text-muted-foreground font-medium"
          >
            Title
          </label>
          <Input
            id="alert-title"
            value={title}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setTitle(e.target.value)
            }
            className="h-8 text-xs"
          />
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="alert-desc"
            className="text-xs text-muted-foreground font-medium"
          >
            Description
          </label>
          <Input
            id="alert-desc"
            value={description}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setDescription(e.target.value)
            }
            className="h-8 text-xs"
          />
        </div>
      </div>
    </PreviewCard>
  );
}

export default function PlaygroundBaseUI() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 overflow-auto p-1">
      <ButtonPreview />
      <InputPreview />
      <DialogPreview />
      <BadgePreview />
      <AlertPreview />
    </div>
  );
}
